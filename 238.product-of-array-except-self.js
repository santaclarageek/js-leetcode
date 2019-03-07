/*
 * @lc app=leetcode id=238 lang=javascript
 *
 * [238] Product of Array Except Self
 */
/**
 * @param {number[]} nums
 * @return {number[]}
 */
var productExceptSelf = function(nums) {
    var fromLeft = [];
    fromLeft[0] = 1;
    for(let i = 1; i < nums.length; i++){
        fromLeft[i] = fromLeft[i-1]*nums[i-1];
    }
    var right = 1;
    for(let i = nums.length - 1; i >= 0; i--){
        fromLeft[i] *= right;
        right *= nums[i];
    }
    return fromLeft;
};

