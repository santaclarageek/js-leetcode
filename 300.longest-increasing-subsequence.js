/*
 * @lc app=leetcode id=300 lang=javascript
 *
 * [300] Longest Increasing Subsequence
 */
/**
 * @param {number[]} nums
 * @return {number}
 */
var lengthOfLIS = function(nums) {
    var arr = [];
    for(let i = 0; i < nums.length; i++){
        arr.push(1);
        for(let j = 0; j < i; j++){
            if(nums[j] < nums[i]) {
                arr[i] = Math.max(arr[i], arr[j] + 1);
            }
        }
    }
    return nums.length ? Math.max(...arr) : 0;
};

