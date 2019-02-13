/*
 * @lc app=leetcode id=203 lang=javascript
 *
 * [203] Remove Linked List Elements
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
 * @param {number} val
 * @return {ListNode}
 */
var removeElements = function(head, val) {
    var actual = head;
    var prev = null;//类似于array中-1位index
    while(actual){
        if(actual.val === val){
            if(!prev){
                head = actual.next;
            }else{
                prev.next = actual.next;
            }
        }else{
            prev = actual;
        }
        actual = actual.next;
    }
    return head;
};

