#![forbid(unsafe_code)]

mod bench;
mod chat;
mod cli;
mod complete;
mod download;

fn main() -> anyhow::Result<()> {
    cli::run()
}
