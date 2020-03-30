#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn _a_works() {
        let _first = A {
            a_1: String::from("Hello"),
            a_2: 1,
            a_3: 2,
        };
    }
}

pub struct A {
    pub a_1: String,
    pub a_2: i16,
    pub a_3: i16,
}

impl A {
    pub fn calculate_a(&self) -> i16 {
        self.a_2 * self.a_3
    }
}
