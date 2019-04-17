/*
 * @lc app=leetcode id=349 lang=javascript
 *
 * [349] Intersection of Two Arrays
 */
/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number[]}
 */
var intersection = function(nums1, nums2) {
    //骚气的操作
    const set = new Set(nums1);
    return [...new Set(nums2.filter(n => (set.has(n))))];
   //这样写也可以 ： return nums2.filter(n => set.delete(n));
};

