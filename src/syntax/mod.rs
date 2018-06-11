pub mod graph;
pub mod transition;

/// Projectivizes a dependency tree
///
/// References:
/// - https://github.com/tensorflow/models/blob/7d30a017fe50b648be6dee544f8059bde52db562/syntaxnet/syntaxnet/document_filters.cc#L296
pub fn projectivize(heads: &[u32]) -> Vec<u32> {
    let mut heads: Vec<i32> = heads.iter().map(|head| *head as i32).collect();
    let num_tokens = heads.len();
    let mut left: Vec<i32> = vec![-1; num_tokens];
    let mut right: Vec<i32> = vec![-1; num_tokens];
    loop {
        for i in 0..num_tokens {
            left[i] = -1;
            right[i] = num_tokens as i32;
        }

        for (i, head) in heads.iter().enumerate() {
            let l = (i as i32).min(*head);
            let r = (i as i32).max(*head);
            for j in (l + 1)..r {
                let j = j as usize;
                if left[j] < l {
                    left[j] = l;
                }
                if right[j] > r {
                    right[j] = r;
                }
            }
        }

        let mut deepest_arc = -1;
        let mut max_depth = 0;
        for (i, head) in heads.iter().enumerate() {
            if *head == 0 {
                continue;
            }
            let l = (i as i32).min(*head);
            let r = (i as i32).max(*head);
            let left_bound = left[l as usize].max(left[r as usize]);
            let right_bound = right[l as usize].min(right[r as usize]);

            if l < left_bound || r > right_bound {
                let mut depth = 0;
                let mut j = i;
                while j != 0 {
                    depth += 1;
                    j = heads[j] as usize;
                }
                if depth > max_depth {
                    deepest_arc = i as i32;
                    max_depth = depth;
                }
            }
        }

        if deepest_arc == -1 {
            return heads.iter().map(|head| *head as u32).collect();
        }

        let lifted_head = heads[heads[deepest_arc as usize] as usize];
        heads[deepest_arc as usize] = lifted_head;
    }
}
