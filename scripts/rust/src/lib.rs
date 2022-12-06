use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::io::BufRead;
use std::path::Path;
use std::rc::Rc;

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
pub fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<fs::File>>>
where
    P: AsRef<Path>,
{
    let file = fs::File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

/// {category: {instance_id: filepath, ...}, ...}}
pub fn calc_categories<P>(meta_data_file: P) -> HashMap<String, Rc<HashSet<String>>>
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
