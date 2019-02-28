/*
 * @lc app=leetcode id=206 lang=javascript
 *
 * [206] Reverse Linked List
 */
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var reverseList = function(head) {
    var tail = null;
    var atucal = head;//这一步可有可无
    var next;
    while(atucal){
        //1.save old node
        next = atucal.next;
        //2.conect new node
        atucal.next = tail;
        //3.update and move pointers
        tail = atucal;
        atucal = next;
    }
    return tail;
};

