use l2::module_abc::module_a::*;

fn main() {
    let first = A {
        a_1: String::from("Hello"),
        a_2: 1,
        a_3: 2,
    };

    println!("{}", first.a_1);
}
