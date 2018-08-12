/*
 * @lc app=leetcode id=2 lang=javascript
 *
 * [2] Add Two Numbers
 */
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
// var addTwoNumbers = function(l1, l2) {
//     var List = new ListNode(0);
//     var head = List;
//     var sum = 0;
//     var carry = 0;

//     while(l1!==null||l2!==null||sum>0){

//         if(l1!==null){
//             sum = sum + l1.val;
//             l1 = l1.next;
//         }
//         if(l2!==null){
//             sum = sum + l2.val;
//             l2 = l2.next;
//         }
//         if(sum>=10){
//             carry = 1;
//             sum = sum - 10;
//         }

//         head.next = new ListNode(sum);
//         head = head.next;

//         sum = carry;
//         carry = 0;

//     }

//     return List.next;
// };

const addTwoNumbers = function(l1, l2) {
    const before = new ListNode();
    let list = before;
    let c = 0;
  
    while (l1 || l2 || c) {
      const v1 = l1 ? l1.val : 0;
      const v2 = l2 ? l2.val : 0;
      const v = v1+v2+c;
  
      list.next = new ListNode(v%10);
      list = list.next;
      c = v >= 10 ? 1 : 0;
      l1 = l1&&l1.next;
      //如果l1存在，就直接取第二个的值；如果不存在，就是null（false）
      //console.log('每轮 ',l1);
      l2 = l2&&l2.next;
    }
  
    return before.next;
  }
