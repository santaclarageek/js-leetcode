/*
 * @lc app=leetcode id=316 lang=javascript
 *
 * [316] Remove Duplicate Letters
 */
/**
 * @param {string} s
 * @return {string}
 */
var removeDuplicateLetters = function(s) {
    let stack = [];
    let counter = Array(26).fill(0);
    let visited = Array(26).fill(false);
    for(let i = 0; i < s.length; i++) {
      counter[s[i].charCodeAt() - 97]++;
    }
    
    for(let i = 0; i < s.length; i++) {
      const c = s[i];
      counter[c.charCodeAt() - 97]--;
      if(visited[c.charCodeAt() - 97]) continue;
      
      while(stack.length && c <= stack[stack.length - 1] && counter[stack[stack.length - 1].charCodeAt() - 97] > 0) {
        visited[stack.pop().charCodeAt() - 97] = false;
      }
      stack.push(c);
      visited[c.charCodeAt() - 97] = true;
    }
    
    return stack.join('');
};

