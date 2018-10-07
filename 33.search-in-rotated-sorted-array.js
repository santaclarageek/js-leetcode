/*
 * @lc app=leetcode id=33 lang=javascript
 *
 * [33] Search in Rotated Sorted Array
 */
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function(nums, target) {
    if(nums === null || nums.length === 0){
        return -1;
    }
    var left = 0;
    var right = nums.length - 1;
    while(left <= right){
        //因为js没有int，要强制转换成int
        //console.log(JSON.stringify(`left: ${left}, right: ${right}`));
        var mid = left + Math.floor((right - left)/2);
        if(nums[mid] === target){
            return mid;
        }else if(nums[mid] < nums[left]){//右边是排序的
            if(target > nums[mid] && target <= nums[right]){
                left = mid + 1;//目标在右边
            }else{
                right = mid - 1;//目标在左边
            }
        }else{//左边是排序的
            if(target < nums[mid] && target >= nums[left]){
                right = mid - 1;//目标在左边
            }else{
                left = mid + 1;//目标在右边
            }
        }
    }
    return -1;
};

