/*
 * @lc app=leetcode id=46 lang=javascript
 *
 * [46] Permutations
 */
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var permute = function(nums) {
    let res = [];
    res.push([]);
    for(let i = 0; i < nums.length; i++){
        let size = res.length;
        for(let j = 0; j < size; j++){
            let list = res.shift();
            for(let k = 0; k <= list.length; k++){
                //spread 写法    
                //在每一个空缺处插入新数值，空缺数(k+1)等于当前数值的原index+1 
                let newList = [...list.slice(0, k), nums[i], ...list.slice(k)];
                res.push(newList);
            }
        }
    }
    return res;   
};

// 李哥DFS版本
// var dfs = (nums, index, res) => {
//     if (index === nums.length) {
//         res.push(nums.slice(0));
//         return;
//     }
//     for (let i = index; i < nums.length; i++) { //destructing
//         [nums[i], nums[index]] = [nums[index], nums[i]];
//         dfs(nums, index + 1, res);
//         [nums[i], nums[index]] = [nums[index], nums[i]];
//     }
// }

// var permute = function(nums) {
// let res = [];
// dfs(nums, 0, res);
// return res;

// }

