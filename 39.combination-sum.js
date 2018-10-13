/*
 * @lc app=leetcode id=39 lang=javascript
 *
 * [39] Combination Sum
 */
/**
 * @param {number[]} candidates
 * @param {number} target
 * @return {number[][]}
 */
const combinationSum = (array, target) => {
    const logic = (array, target, index, set = []) => {//set = [] 是default
      if (target === 0) {
        result.push(set);
      }
      for (let i = index; i < array.length; i++) {
        if (target - array[i] >= 0) {
          logic(array, target - array[i], i, set.concat(array[i]));
        }
      }
    };
    const set = [];
    const result = [];//array是object，refrence不会变
    logic(array, target, 0, set);
    return result;
};

