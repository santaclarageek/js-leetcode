/*
 * @lc app=leetcode id=283 lang=javascript
 *
 * [283] Move Zeroes
 */
/**
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
 */
var moveZeroes = function(nums) {
    let numberOfZeroes = 0;
    nums.forEach((num, idx) => {
        while(nums[idx] === 0) {//把为0的删掉
            nums.splice(idx, 1);
            numberOfZeroes++;
        }
    });
    for (let z = 0; z < numberOfZeroes; z++) {
        nums.push(0);
    }
    return nums;
};

