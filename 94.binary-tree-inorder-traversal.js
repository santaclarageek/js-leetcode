/*
 * @lc app=leetcode id=94 lang=javascript
 *
 * [94] Binary Tree Inorder Traversal
 */
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */

 var inorderTraversal = function(root) {
    var res = [];
    if(root){//这里一定要判断root是否存在
        helper(root,res);
    }        
    return res;
};

var helper = function(root,res) {
    if(root.left){
        helper(root.left,res);
    }
    res.push(root.val);//push的操作就留给总结，不要push（helper）
    if(root.right){
        helper(root.right,res);
    }
};

