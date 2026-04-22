#![forbid(unsafe_code)]

mod bench;
mod chat;
mod cli;
mod complete;
mod download;
mod eval;
mod quantize;

fn main() -> anyhow::Result<()> {
    cli::run()
}
