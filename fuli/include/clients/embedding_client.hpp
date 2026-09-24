#pragma once

#include <boost/asio/awaitable.hpp>
#include <boost/asio/io_context.hpp>
#include <string>
#include <vector>

namespace clients {

// Async HTTP client for the HF text-embeddings-inference container
// (bge-m3, native /embed route — not the OpenAI-compatible one).
class EmbeddingClient {
public:
  EmbeddingClient(boost::asio::io_context &ioc, std::string host,
                   std::string port);

  boost::asio::awaitable<std::vector<float>> Embed(std::string text) const;

private:
  boost::asio::io_context &ioc_;
  std::string host_;
  std::string port_;
};

} // namespace clients
