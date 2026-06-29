mod hub;
mod routing;

use tokio::net::TcpListener;
use tokio_tungstenite::accept_async;
use tracing::{error, info};

#[tokio::main]
async fn main() {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "delta_edge=debug,info".parse().unwrap()),
        )
        .init();

    let address: String = std::env::var("EDGE_BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    let listener: TcpListener = TcpListener::bind(&address).await.expect("Failed to bind");

    info!("delta-edge listening on {address}");

    let hub = hub::Hub::new();

    while let Ok((stream, peer)) = listener.accept().await {
        let hub = hub.clone();
        tokio::spawn(async move {
            match accept_async(stream).await {
                Ok(ws) => {
                    info!("[{peer}] WebSocket connected");
                    hub.handle_connection(ws, peer).await;
                }
                Err(e) => error!("[{peer}] Handshake failed: {e}"),
            }
        });
    }
}
