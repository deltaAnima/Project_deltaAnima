#include "clients/redis_db_client.hpp"

#include <boost/asio/consign.hpp>
#include <boost/asio/detached.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/redis/connection.hpp>
// boost::redis uses Boost's usual "separate compilation" pattern:
// src.hpp pulls in the .ipp files that actually DEFINE connection's
// methods, request::push, resp3 serialization, etc. It must be included
// in exactly one .cpp across the whole program — this is that one,
// since redis_db_client.cpp is currently the only boost::redis consumer.
// Without this, everything above compiles fine (the declarations are
// all in the headers) but the linker can't find any of the symbols.
#include <boost/redis/src.hpp>
#include <chrono>
#include <map>

#include "Third_Party/json.hpp"

namespace clients {

namespace net = boost::asio;
namespace redis = boost::redis;
using json = nlohmann::json;

namespace {

std::string MemKey(int64_t faiss_id) { return "mem:" + std::to_string(faiss_id); }

// HGETALL comes back as a flat {string: string} map — these two helpers
// are the only places that know MemoryMetadata's nested C++ shape maps
// onto dot-prefixed flat field names ("memory.content.user_input", ...)
// in Redis. Everything else (Get/SetMemoryMetadata) just moves data
// through them.
std::string Get(const std::map<std::string, std::string> &fields,
                 const std::string &key, std::string def = "") {
  auto it = fields.find(key);
  return it == fields.end() ? def : it->second;
}

float GetFloat(const std::map<std::string, std::string> &fields,
               const std::string &key, float def) {
  auto it = fields.find(key);
  return it == fields.end() ? def : std::stof(it->second);
}

int64_t GetInt64(const std::map<std::string, std::string> &fields,
                  const std::string &key, int64_t def) {
  auto it = fields.find(key);
  return it == fields.end() ? def : std::stoll(it->second);
}

MemoryMetadata FieldsToMetadata(const std::map<std::string, std::string> &f) 
{
  MemoryMetadata meta;

  meta.memory.content.user_input = Get(f, "memory.content.user_input");
  meta.memory.content.model_response = Get(f, "memory.content.model_response");

  meta.memory.user.user_id.high =
      static_cast<uint64_t>(GetInt64(f, "memory.user.user_id.high", 0));
  meta.memory.user.user_id.low =
      static_cast<uint64_t>(GetInt64(f, "memory.user.user_id.low", 0));
  meta.memory.user.user_name = Get(f, "memory.user.user_name");
  meta.memory.user.user_content = Get(f, "memory.user.user_content");

  meta.memory.persona.persona_name =
      Get(f, "memory.persona.persona_name", "Reminh");
  meta.memory.persona.persona_content = Get(f, "memory.persona.persona_content");

  // emotion_terms is stored as a JSON array string (Redis hash values
  // are plain strings, so a std::vector<std::string> has to be encoded
  // somehow — a JSON array is the least surprising choice given every
  // other structured value in this project already goes through
  // nlohmann::json).
  std::string terms_json = Get(f, "memory.emotion.emotion_terms", "[]");
  json parsed_terms = json::parse(terms_json, nullptr, /*allow_exceptions=*/false);

  if (!parsed_terms.is_discarded() && parsed_terms.is_array())
    meta.memory.emotion.emotion_terms =
        parsed_terms.get<std::vector<std::string>>();

  meta.memory.emotion.intensity = GetFloat(f, "memory.emotion.intensity", 0.0f);

  meta.emotion_analysis.deltaEGO_analysis = Get(f, "emotion_analysis.deltaEGO_analysis", "{}");

  std::string session_id = Get(f, "metadata.session_id");

  if (!session_id.empty())
    meta.metadata.session_id = session_id;

  meta.metadata.importance = GetFloat(f, "metadata.importance", 0.5f);
  meta.metadata.timestamp = GetInt64(f, "metadata.timestamp", 0);
  meta.metadata.faiss_id = GetInt64(f, "metadata.faiss_id", 0);

  std::string request_json = Get(f, "query.request_query", "{}");
  json parsed_request = json::parse(request_json, nullptr, /*allow_exceptions=*/false);
  if (!parsed_request.is_discarded()) 
  {
    try 
    {
      meta.query.request_query = parsed_request.get<schemas::FuliContextRequest>();
    } 
    catch (...) 
    {
      // Leave query.request_query default-constructed if it doesn't
      // parse — a malformed/missing stored request shouldn't fail the
      // whole GetMemoryMetadata call.
    }
  }

  return meta;
}

std::map<std::string, std::string> MetadataToFields(const MemoryMetadata &meta) 
{
  return {
      {"memory.content.user_input", meta.memory.content.user_input},
      {"memory.content.model_response", meta.memory.content.model_response},
      {"memory.user.user_id.high", std::to_string(meta.memory.user.user_id.high)},
      {"memory.user.user_id.low", std::to_string(meta.memory.user.user_id.low)},
      {"memory.user.user_name", meta.memory.user.user_name},
      {"memory.user.user_content", meta.memory.user.user_content},
      {"memory.persona.persona_name", meta.memory.persona.persona_name},
      {"memory.persona.persona_content", meta.memory.persona.persona_content},
      {"memory.emotion.emotion_terms", json(meta.memory.emotion.emotion_terms).dump()},
      {"memory.emotion.intensity", std::to_string(meta.memory.emotion.intensity)},
      {"emotion_analysis.deltaEGO_analysis", meta.emotion_analysis.deltaEGO_analysis},
      {"metadata.session_id", meta.metadata.session_id.value_or("")},
      {"metadata.importance", std::to_string(meta.metadata.importance)},
      {"metadata.timestamp", std::to_string(meta.metadata.timestamp)},
      {"metadata.faiss_id", std::to_string(meta.metadata.faiss_id)},
      {"query.request_query", json(meta.query.request_query).dump()},
  };
}

} // namespace

RedisDbClient::RedisDbClient(net::io_context &ioc, std::string host,
                              std::string port)
    : ioc_(ioc), host_(std::move(host)), port_(std::move(port)),
      conn_(std::make_shared<redis::connection>(ioc)) {}

// Defined here (not defaulted inline in the header) for the same reason
// as every other Pimpl-shaped destructor in this project: conn_'s
// pointee type (boost::redis::connection) is only forward-declared in
// the header, so std::shared_ptr's deleter needs the complete type,
// which <boost/redis/connection.hpp> only provides here.
RedisDbClient::~RedisDbClient() = default;

void RedisDbClient::Run() 
{
  redis::config cfg;
  cfg.addr.host = host_;
  cfg.addr.port = port_;

  // consign keeps conn_ (the shared_ptr) alive for as long as this
  // background run operation is active. async_run itself never
  // completes on its own — it keeps the connection open/reconnecting
  // until cancel() is called — so this is meant to run for the whole
  // server's lifetime, same as GpuFaissEngine's worker thread.
  conn_->async_run(cfg, {}, net::consign(net::detached, conn_));
}

net::awaitable<std::optional<MemoryMetadata>>
RedisDbClient::GetMemoryMetadata(int64_t faiss_id)
{
  redis::request req;
  req.push("HGETALL", MemKey(faiss_id));

  redis::response<std::map<std::string, std::string>> resp;
  co_await conn_->async_exec(req, resp, net::use_awaitable);

  const auto &fields = std::get<0>(resp).value();
  if (fields.empty())
    co_return std::nullopt; // HGETALL on a missing key comes back empty

  co_return FieldsToMetadata(fields);
}

net::awaitable<void>
RedisDbClient::SetMemoryMetadata(int64_t faiss_id, const MemoryMetadata &meta)
{
  redis::request req;
  req.push_range("HSET", MemKey(faiss_id), MetadataToFields(meta));

  co_await conn_->async_exec(req, redis::ignore, net::use_awaitable);
}

net::awaitable<int64_t> RedisDbClient::NextId()
{
  redis::request req;
  req.push("INCR", "next_mem_id");

  redis::response<int64_t> resp;
  co_await conn_->async_exec(req, resp, net::use_awaitable);

  co_return std::get<0>(resp).value();
}

nlohmann::json RedisDbClient::HealthCheck() const
{
  if (host_.empty() || port_.empty()) 
  {
    return {{"status", "error"},
            {"message", "Class RedisDbClient -> host or port is empty"}};
  }

  try 
  {
    // Throwaway io_context + connection, entirely separate from ioc_/
    // conn_ — same "self-contained blocking check" shape as
    // EmbeddingClient::HealthCheck, so HealthChecker::CheckAll can
    // std::async both the same way without either one touching the
    // server's real connection/executor.
    net::io_context local_ioc;
    auto local_conn = std::make_shared<redis::connection>(local_ioc);

    redis::config cfg;
    cfg.addr.host = host_;
    cfg.addr.port = port_;
    local_conn->async_run(cfg, {}, [](boost::system::error_code) {});

    redis::request req;
    req.push("PING");
    redis::response<std::string> resp;

    boost::system::error_code exec_ec;
    bool timed_out = false;

    net::steady_timer timer(local_ioc);
    timer.expires_after(std::chrono::seconds(5));
    timer.async_wait([&](boost::system::error_code ec) 
    {
      if (!ec) 
      {
        timed_out = true;
        local_conn->cancel(); // unstick everything below if PING hangs
      }
    });

    local_conn->async_exec(
        req, resp, [&](boost::system::error_code ec, std::size_t) 
        {
          exec_ec = ec;
          timer.cancel();
          local_conn->cancel(); // stop async_run so local_ioc.run() can return
        });

    local_ioc.run();

    if (timed_out) 
    {
      return {{"status", "unhealthy"},
              {"message", "Class RedisDbClient -> PING timed out"}};
    }
    if (exec_ec) 
    {
      return {{"status", "unhealthy"},
              {"message", "Class RedisDbClient -> " + exec_ec.message()}};
    }

    return {{"status", "healthy"},
            {"message", "Class RedisDbClient -> PING ok: " +
                            std::get<0>(resp).value()}};
  } 
  catch (const std::exception &e) 
  {
    return {{"status", "error"},
            {"message", std::string(
                            "Class RedisDbClient -> Exception during health "
                            "check: ") +
                            e.what()}};
  }
}

} // namespace clients
