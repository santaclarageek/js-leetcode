/*
 * @lc app=leetcode id=394 lang=javascript
 *
 * [394] Decode String
 */
/**
 * @param {string} s
 * @return {string}
 */
var decodeString = function(s) {
    // //(\d+)? 指的是数字部分不一定是必须的
    // myReg = /(\d+)?\[([a-zA-Z]+)\]/;
    // //用while loop来一层层扒皮解套
    // while(myReg.test(s)) {
    //     s = s.replace(myReg, (full, num, str) => {
    //         //没有num，就默认str只出现一次；有，就按num次数来repeat str
    //         //(full, num, str) 是replacer function，在这里他们是对应的(match, p1, p2)
    //         //https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/replace
    //         return num === undefined ? str : str.repeat(num);
    //     });
    // }
    // return s;
    
    var myReg = /(\d+)?\[([a-zA-Z]+)\]/;
    while(myReg.test(s)){
        s = s.replace(myReg, (match, num, str) => {
            return num === undefined ? str : str.repeat(num);
        })
    }
    return s;
};

