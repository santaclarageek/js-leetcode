/*
 * @lc app=leetcode id=151 lang=javascript
 *
 * [151] Reverse Words in a String
 */
/**
 * @param {string} s
 * @return {string}
 */
var reverseWords = function(s) {
    let result = [];
    let arr = s.split(" ").filter(el => !(el === ""));
    for (s of arr) {
        result.unshift(s);
    }
    return result.join(" ");
};

