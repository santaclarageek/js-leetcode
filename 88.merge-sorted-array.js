/*
 * @lc app=leetcode id=88 lang=javascript
 *
 * [88] Merge Sorted Array
 */
/**
 * @param {number[]} nums1
 * @param {number} m
 * @param {number[]} nums2
 * @param {number} n
 * @return {void} Do not return anything, modify nums1 in-place instead.
 */
   	//直观思路是双指针i, j同时扫描A, B，选min(A[i], B[j])作为下一个元素插入;
	//但是只能利用A后面的空间来插入，这样就很不方便
	//反向思路，merge后的数组一共有m+n个数;
	//i, j从A, B尾部扫描，选max(A[i], B[j])插入从m+n起的尾部!
	//这样也可以防止插入到A原来数字的范围内时，overwrite掉A原来的数。

    var merge = function(nums1, m, nums2, n) {
        let i = m - 1, j = n - 1, index = n + m - 1;
        while(i >= 0 && j >= 0){
            nums1[index--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--]; 
        }
        while(j >= 0){//如果i先走到头
            nums1[index--] = nums2[j--];
        }
    };
      
