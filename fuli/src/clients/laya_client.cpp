#include "clients/laya_client.hpp"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/asio/use_future.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace clients {

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

namespace {

// Covers the whole connect + write + read exchange. Laya answers in tens
// of ms on GPU (a few hundred on CPU), so anything near this is a dead or
// overloaded server, not a slow answer.
constexpr auto kLayaTimeoutSeconds = std::chrono::seconds(5);

// TODO(laya): known weaknesses of the wording below, vs the 5090's answers:
//  - Positive messages come out compressed: "I love you Reminh" scores
//    +0.33 valence (5090: +0.80), "WE WON THE CHAMPIONSHIP!!!" +0.38
//    (5090: +0.80). Strong joy/affection barely lifts the mood.
//  - Ordinary questions/greetings sit slightly negative on dominance
//    (around -0.2) instead of at 0.
//  - The wording was tuned against only 22 messages, with the 5090 as the
//    reference rather than ground truth — re-check on a larger set before
//    tuning further, or it will overfit those 22.
// Ordered lowest -> highest: Laya's `score` is the expected index into
// this list. Seven levels gives the -1..1 mapping a 1/3 step per level.
// The parenthesized examples anchor each level — with bare labels
// ("neutral", "slightly negative", ...) Laya scored plain questions and
// greetings around -0.4 valence, which would drag every mood negative.
const std::vector<std::string> kValenceLevels = {
    "very negative (insulted, threatened, heartbroken)",
    "negative (hurt, annoyed)",
    "slightly negative (mildly uneasy)",
    "neutral (ordinary question, greeting, small talk)",
    "slightly positive (pleasantly greeted, thanked)",
    "positive (praised, happy news)",
    "very positive (loved, overjoyed)"};
const std::vector<std::string> kArousalLevels = {
    "very calm (sleepy, peaceful)",
    "calm (relaxed, sad, tired)",
    "slightly calm",
    "neutral (ordinary question, greeting, small talk)",
    "slightly excited",
    "excited (startled, angry, thrilled)",
    "very excited (panic, rage, ecstatic shouting)"};
const std::vector<std::string> kDominanceLevels = {
    "very controlled (threatened, ordered around)",
    "controlled (pressured, begged)",
    "slightly controlled",
    "neutral (ordinary question, greeting, small talk)",
    "slightly in control (asked for help)",
    "in control (praised, relied on)",
    "very in control (fully confident, in charge)"};

// Asked from the LISTENER's side — how hearing the message would feel —
// to match the 5090 prompt (how Reminh feels, not the speaker's tone).
// Asked about the speaker instead, Laya rates a command like "Do it now"
// as dominant, the opposite sign of what the listener's mood should get.
// Spelling out what pushes each axis up or down mattered most for
// arousal: with a one-line question Laya rated even "I'm so furious, get
// out!" as calm. Tuned against the 5090's answers on a 22-message set.
constexpr const char *kValenceInstructions =
    "You are the listener. Someone just said this message to you. Rate how pleasant "
    "or unpleasant hearing it makes you feel. Praise, affection, good news and "
    "gratitude feel positive. Insults, threats, rejection and bad news feel negative. "
    "Ordinary questions, requests for information, greetings, thanks and small talk "
    "leave you neutral.";
constexpr const char *kArousalInstructions =
    "You are the listener. Someone just said this message to you. Rate how activated "
    "or energized hearing it makes you feel. Shouting, anger, panic, fear, big "
    "excitement and urgent demands are highly exciting. Sadness, tiredness and "
    "peaceful scenes are calm. Ordinary questions, greetings and small talk leave you "
    "neutral.";
constexpr const char *kDominanceInstructions =
    "You are the listener. Someone just said this message to you. Rate how much in "
    "control you feel after hearing it. Orders, insults, threats and pleading put "
    "pressure on you and make you feel controlled. Praise, requests for your help or "
    "knowledge, and kind words make you feel in control. Ordinary questions, greetings "
    "and small talk leave you neutral.";

json ScoreQuestion(const char *instructions, const std::vector<std::string> &levels)
{
  return {{"type", "score"}, {"instructions", instructions}, {"criteria", levels}};
}

// Expected ordinal level 0..levels-1 -> -1..1.
float ToUnitRange(const json &answers, const char *qid, std::size_t levels)
{
  double score = answers.at(qid).at("score").get<double>();
  double unit = 2.0 * score / static_cast<double>(levels - 1) - 1.0;
  return std::clamp(static_cast<float>(unit), -1.0f, 1.0f);
}

// One request per connection, like every other client here. The deadline
// is set once and spans every async step below; beast cancels whichever
// one is pending when it expires and that step throws.
net::awaitable<http::response<http::string_body>>
SendRequest(std::string host, std::string port, http::request<http::string_body> req)
{
  auto executor = co_await net::this_coro::executor;

  tcp::resolver resolver(executor);
  beast::tcp_stream stream(executor);
  stream.expires_after(kLayaTimeoutSeconds);

  auto endpoints = co_await resolver.async_resolve(host, port, net::use_awaitable);
  co_await stream.async_connect(endpoints, net::use_awaitable);

  req.set(http::field::host, host);
  req.set(http::field::user_agent, "fuli-orchestrator");
  req.prepare_payload();
  co_await http::async_write(stream, req, net::use_awaitable);

  beast::flat_buffer buffer;
  http::response<http::string_body> res;
  co_await http::async_read(stream, buffer, res, net::use_awaitable);

  beast::error_code ec;
  stream.socket().shutdown(tcp::socket::shutdown_both, ec);
  co_return res;
}

} // namespace

LayaClient::LayaClient(net::io_context &ioc, std::string host, std::string port,
                       std::string api_key)
    : ioc_(ioc), host_(std::move(host)), port_(std::move(port)),
      api_key_(std::move(api_key)) {}

net::awaitable<VadEstimate> LayaClient::EstimateVad(std::string text) const
{
  json body{
      {"state", {{"body", text}}},
      {"questions",
       {
           {"valence", ScoreQuestion(kValenceInstructions, kValenceLevels)},
           {"arousal", ScoreQuestion(kArousalInstructions, kArousalLevels)},
           {"dominance", ScoreQuestion(kDominanceInstructions, kDominanceLevels)},
       }},
  };

  http::request<http::string_body> req{http::verb::post, "/v1/systemone", 11};
  req.set(http::field::content_type, "application/json");
  if (!api_key_.empty())
    req.set(http::field::authorization, "Bearer " + api_key_);
  req.body() = body.dump();

  http::response<http::string_body> res = co_await SendRequest(host_, port_, std::move(req));

  if (res.result() != http::status::ok)
  {
    throw std::runtime_error("Laya /v1/systemone failed: HTTP " +
                             std::to_string(res.result_int()) + " " + res.body());
  }

  json parsed = json::parse(res.body());
  if (!parsed.contains("answers") || !parsed.at("answers").is_object())
    throw std::runtime_error("unexpected Laya response shape: " + res.body());

  const json &answers = parsed.at("answers");
  VadEstimate estimate;
  estimate.v = ToUnitRange(answers, "valence", kValenceLevels.size());
  estimate.a = ToUnitRange(answers, "arousal", kArousalLevels.size());
  estimate.d = ToUnitRange(answers, "dominance", kDominanceLevels.size());
  co_return estimate;
}

nlohmann::json LayaClient::HealthCheck() const
{
  if (host_.empty() || port_.empty())
  {
    return {{"status", "error"}, {"message", "Class LayaClient -> host or port is empty"}};
  }

  try
  {
    // Same coroutine path as EstimateVad, driven on a private io_context:
    // beast's expires_after only applies to async operations, so this is
    // what actually makes the deadline hold here.
    net::io_context local_ioc;
    http::request<http::string_body> req{http::verb::get, "/ready", 11};
    auto future = net::co_spawn(local_ioc, SendRequest(host_, port_, std::move(req)),
                                net::use_future);
    local_ioc.run();
    http::response<http::string_body> res = future.get();

    // /ready: 200 once the configured checkpoints have loaded.
    if (res.result() == http::status::ok)
    {
      return {{"status", "healthy"},
              {"message", "Class LayaClient -> GET /ready: " + res.body()}};
    }
    return {{"status", "unhealthy"},
            {"message", "Class LayaClient -> GET /ready returned HTTP " +
                            std::to_string(res.result_int()) + ": " + res.body()}};
  }
  catch (const std::exception &e)
  {
    return {{"status", "unhealthy"}, {"message", std::string("Class LayaClient -> ") + e.what()}};
  }
}

} // namespace clients
