/// Finds a dependency tree by a greedy spanning tree algorithm
///
/// The tree is wellformed by iteratively identifying and fixing cycles for each proposed root
/// and selecting the one with the highest score.
///
/// References:
/// - http://aclweb.org/anthology/K17-3002
/// - https://github.com/tdozat/Parser-v2/blob/6229befd7ab72565569d9f8aaa98401e8112971d/parser/misc/mst.py
///
/// scores: 2D array [dependents, heads]
pub fn simple_spanning_tree<V: AsRef<[f32]>>(scores: &[V]) -> Vec<usize> {
    // TODO(chantera): implement
    let _ = scores;
    vec![]
}

/// Finds a maximum spanning dependency tree by Chu–Liu/Edmonds' algorithm
///
/// scores: 2D array [dependents, heads]
pub fn chu_liu_edmonds<V: AsRef<[f32]>>(scores: &[V]) -> Vec<usize> {
    // TODO(chantera): implement
    let _ = scores;
    vec![]
}
