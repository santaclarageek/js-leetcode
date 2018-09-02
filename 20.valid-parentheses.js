/*
 * @lc app=leetcode id=20 lang=javascript
 *
 * [20] Valid Parentheses
 */
/**
 * @param {string} s
 * @return {boolean}
 */
var isValid = function(s) {
    let map = { //这个是object
        ")": "(",
        "]": "[",
        "}": "{"
    }
    //map = {} //这个也是object
    //map = new Map(); //这样才有map的特性，空的时候这只是reference
    let arr = [];
    for(let i = 0; i < s.length; i ++){
        if(s[i] === "(" || s[i] === "[" || s[i] === "{"){
            arr.push(s[i]);
        }
        else{
            if(arr[arr.length - 1] === map[s[i]]){
                arr.pop();
            }
            else return false;
        }
    }
    return arr.length === 0 ? true : false;
};

