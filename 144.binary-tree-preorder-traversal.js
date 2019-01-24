/*
 * @lc app=leetcode id=144 lang=javascript
 *
 * [144] Binary Tree Preorder Traversal
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

var preorderTraversal = function(root) {
    
    var res = [];
    if(!root) return res;
    function helper(root){
        if(!root) return;
        res.push(root.val);
        if(root.left){
            helper(root.left);
        }
        
        if(root.right){
            helper(root.right);
        }
        
    }
    helper(root);
    return res;
};

