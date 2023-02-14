use clap::Parser;
use rust::read_lines;
use std::fs;
use std::io::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to original meta dataset file
    #[arg(short, long)]
    src_file: std::path::PathBuf,
    /// The output file path for writing extracted meta dataset file
    #[arg(short, long)]
    dst_file: std::path::PathBuf,
    /// verbose
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    if let Ok(lines) = read_lines(args.src_file) {
        let mut buffer = fs::File::create(args.dst_file).unwrap();
        for line in lines {
            let line = line.unwrap();
            buffer.write_fmt(format_args!("{}\n", &line[17..])).unwrap();
        }
    }
}
