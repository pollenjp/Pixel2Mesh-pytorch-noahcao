use clap::Parser;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::io::prelude::*;
use std::io::BufRead;
use std::path::Path;
use std::rc::Rc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to original meta dataset file
    #[arg(short, long)]
    meta_data_file: std::path::PathBuf,
    /// The output file path for writing extracted meta dataset file
    #[arg(short, long)]
    out_file: std::path::PathBuf,
    /// verbose
    #[arg(short, long)]
    verbose: bool,
}

fn calc_categories<P>(meta_data_file: P) -> HashMap<String, Rc<HashSet<String>>>
where
    P: AsRef<Path>,
{
    let mut categories: HashMap<String, Rc<HashSet<String>>> = HashMap::new();

    if let Ok(lines) = read_lines(meta_data_file) {
        for line in lines {
            if let Ok(line) = line {
                let p = Path::new(&line);
                let instance_id_dir = p.parent().unwrap().parent().unwrap();
                let instance_id = instance_id_dir.file_name().unwrap().to_str().unwrap();
                let class_id = instance_id_dir
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap();

                let mut st: HashSet<String>;
                if categories.contains_key(class_id) {
                    let mut st_rc = categories.get(class_id).unwrap().clone();
                    if !st_rc.contains(instance_id) {
                        categories.remove(class_id);
                        Rc::get_mut(&mut st_rc)
                            .unwrap()
                            .insert(instance_id.to_string());
                        categories.insert(class_id.to_string(), st_rc);
                    }
                } else {
                    st = HashSet::new();
                    st.insert(instance_id.to_string());
                    categories.insert(class_id.to_string(), Rc::new(st));
                }
            }
        }
    }

    categories
}

fn main() {
    let args = Args::parse();

    let categories = calc_categories(&args.meta_data_file);
    // dbg!(&categories);

    // instances ration per class should be same as original
    // 1/400
    // 1/800
    // let ratio: f64 = 1.0 / 800.0;
    let ratio: f64 = 1.0 / 400.0;
    assert!(ratio > 0.0 && ratio < 1.0);

    let mut counter: HashMap<String, Rc<HashSet<String>>> = HashMap::new();

    if let Ok(lines) = read_lines(&args.meta_data_file) {
        let mut buffer = fs::File::create(args.out_file).unwrap();
        for line in lines {
            if let Ok(line) = line {
                let p = Path::new(&line);
                let instance_id_dir = p.parent().unwrap().parent().unwrap();
                let instance_id = instance_id_dir.file_name().unwrap().to_str().unwrap();
                let class_id = instance_id_dir
                    .parent()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap();

                let num_inst = categories.get(class_id).unwrap().len();
                let n_max = (num_inst as f64 * ratio).ceil() as usize;
                if n_max == 0 {
                    continue;
                }

                if counter.contains_key(class_id) {
                    let mut st_rc = counter.get(class_id).unwrap().clone();
                    if !st_rc.contains(instance_id) {
                        if st_rc.len() < n_max {
                            counter.remove(class_id);
                            Rc::get_mut(&mut st_rc)
                                .unwrap()
                                .insert(instance_id.to_string());
                            counter.insert(class_id.to_string(), st_rc);
                        } else {
                            continue;
                        }
                    }
                    buffer.write_fmt(format_args!("{}\n", line)).unwrap();
                } else {
                    let mut st = HashSet::new();
                    st.insert(instance_id.to_string());
                    counter.insert(class_id.to_string(), Rc::new(st));
                    buffer.write_fmt(format_args!("{}\n", line)).unwrap();
                }
            }
        }
    }
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<fs::File>>>
where
    P: AsRef<Path>,
{
    let file = fs::File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
