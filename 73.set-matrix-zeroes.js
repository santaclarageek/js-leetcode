/*
 * @lc app=leetcode id=73 lang=javascript
 *
 * [73] Set Matrix Zeroes
 */
/**
 * @param {number[][]} matrix
 * @return {void} Do not return anything, modify matrix in-place instead.
 */

//最优解，思路https://www.youtube.com/watch?v=g0eBwXJt6Ag
var setZeroes = function(matrix) {
    let isCol = 0;
    let rl = matrix.length;
    let cl = matrix[0].length;
    let i;
    let j;
    for (i = 0; i < rl; i++) {
      if (matrix[i][0] === 0) {//如果第一列有0，用iscol记下
        isCol = 1;
      }
      for (j = 1; j < cl; j++) {//如果某rc有0，在对应的第一行与第一列mark下0
          //用mark down在第一列和第一行的0来判断
        if (matrix[i][j] === 0) {
          matrix[i][0] = 0;
          matrix[0][j] = 0;
        }
      }
    }
    for (i = rl - 1; i >= 0; i--) {
      for (j = cl - 1; j >= 1; j--) {
        if (matrix[i][0] === 0 || matrix[0][j] === 0) {//如果对应的r/c为零，更改当前为0
          matrix[i][j] = 0;
        }
      }
      if (isCol) {//如果第一col有0，把第一行设为零
        matrix[i][0] = 0;
      }
    }
  };

