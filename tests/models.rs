extern crate monolith;
extern crate primitiv;

#[cfg(feature = "models")]
mod tests {
    use monolith::models::*;
    use monolith::utils::primitiv as primitiv_utils;
    use primitiv::functions as F;
    use primitiv::*;

    // #[test]
    // #[should_panic]
    // fn test_bilinear_tensor() {
    //     let mut dev = primitiv_utils::select_device(-1);
    //     devices::set_default(&mut *dev);
    //     let mut bilinear = Bilinear::new((true, true, true));
    //     bilinear.init(100, 300, 128);
    //     let t1: Tensor = F::random::normal(([100, 150, 80], 32), 0.0, 1.0);
    //     let t2: Tensor = F::random::normal(([300, 150, 60], 32), 0.0, 1.0);
    //     bilinear.forward(t1, t2);
    // }

    #[test]
    fn test_bilinear() {
        let mut dev = primitiv_utils::select_device(-1);
        devices::set_default(&mut *dev);
        {
            let mut bilinears = vec![
                Bilinear::new((true, true, true)),
                Bilinear::new((true, true, false)),
                Bilinear::new((true, false, true)),
                Bilinear::new((false, true, true)),
                Bilinear::new((true, false, false)),
                Bilinear::new((false, true, false)),
                Bilinear::new((false, false, true)),
                Bilinear::new((false, false, false)),
            ];
            for bilinear in &mut bilinears {
                bilinear.init(100, 300, 128);
                // matrix, martix
                let t1: Tensor = F::random::normal(([100, 80], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300, 60], 32), 0.0, 1.0);
                let y = bilinear.forward(t1, t2);
                let s = y.shape();
                assert!(s.dims().len() == 3);
                assert!(s.at(0) == 80);
                assert!(s.at(1) == 128);
                assert!(s.at(2) == 60);
                assert!(s.batch() == 32);
                // matrix, vector
                let t1: Tensor = F::random::normal(([100, 80], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300], 32), 0.0, 1.0);
                let y = bilinear.forward(&t1, &t2);
                let s = y.shape();
                assert!(s.dims().len() == 2);
                assert!(s.at(0) == 80);
                assert!(s.at(1) == 128);
                assert!(s.batch() == 32);
                // vector, matrix
                let t1: Tensor = F::random::normal(([100], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300, 60], 32), 0.0, 1.0);
                let y = bilinear.forward(&t1, &t2);
                let s = y.shape();
                assert!(s.dims().len() == 2);
                assert!(s.at(0) == 128);
                assert!(s.at(1) == 60);
                assert!(s.batch() == 32);
                // vector, vector
                let t1: Tensor = F::random::normal(([100], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300], 32), 0.0, 1.0);
                let y = bilinear.forward(&t1, &t2);
                let s = y.shape();
                assert!(s.dims().len() == 1);
                assert!(s.at(0) == 128);
                assert!(s.batch() == 32);
                // matrix[out=1], matrix[out=1]
                let y = bilinear.forward(F::reshape(t1, [100, 1]), F::reshape(t2, [300, 1]));
                let s = y.shape();
                assert!(s.dims().len() == 1);
                assert!(s.at(0) == 128);
                assert!(s.batch() == 32);
            }
        }
        {
            let mut bilinears = vec![
                Bilinear::new((true, true, true)),
                Bilinear::new((true, true, false)),
                Bilinear::new((true, false, true)),
                Bilinear::new((false, true, true)),
                Bilinear::new((true, false, false)),
                Bilinear::new((false, true, false)),
                Bilinear::new((false, false, true)),
                Bilinear::new((false, false, false)),
            ];
            for bilinear in &mut bilinears {
                bilinear.init(100, 300, 1);
                // matrix, martix
                let t1: Tensor = F::random::normal(([100, 80], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300, 60], 32), 0.0, 1.0);
                let y = bilinear.forward(t1, t2);
                let s = y.shape();
                assert!(s.dims().len() == 2);
                assert!(s.at(0) == 80);
                assert!(s.at(1) == 60);
                assert!(s.batch() == 32);
                // matrix, vector
                let t1: Tensor = F::random::normal(([100, 80], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300], 32), 0.0, 1.0);
                let y = bilinear.forward(&t1, &t2);
                let s = y.shape();
                assert!(s.dims().len() == 1);
                assert!(s.at(0) == 80);
                assert!(s.batch() == 32);
                // vector, matrix
                let t1: Tensor = F::random::normal(([100], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300, 60], 32), 0.0, 1.0);
                let y = bilinear.forward(&t1, &t2);
                let s = y.shape();
                assert!(s.dims().len() == 1);
                assert!(s.at(0) == 60);
                assert!(s.batch() == 32);
                // vector, vector
                let t1: Tensor = F::random::normal(([100], 32), 0.0, 1.0);
                let t2: Tensor = F::random::normal(([300], 32), 0.0, 1.0);
                let y = bilinear.forward(&t1, &t2);
                let s = y.shape();
                assert!(s.dims().len() == 0);
                assert!(s.batch() == 32);
                // matrix[out=1], matrix[out=1]
                let y = bilinear.forward(F::reshape(t1, [100, 1]), F::reshape(t2, [300, 1]));
                let s = y.shape();
                assert!(s.dims().len() == 0);
                assert!(s.batch() == 32);
            }
        }
    }
}
