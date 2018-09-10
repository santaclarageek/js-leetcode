process.stdin.resume();
process.stdin.setEncoding('utf8');
const fs = require('fs');
var stdin = '';


process.stdin.on('data', function (chunk) {
  stdin += chunk;
  
}).on('end', function() {
  var lines = stdin.trim().split('\n');
  
  for(var i=0; i<lines.length; i++) {
    process.stdout.write(coinChange(parseInt(lines[i])).toString());
  }
});


const wordToNumber = word => {
  if (word === "zero") {
    return 0;
  }
  const units = ["thousand", "million", "billion"];
  const nums = [
    "",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen"
  ];
  const tens = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety"
  ];
  if (word === null) {
      return 0;
  }
  const words = word.split(" ");
  
  let res = "";
  for (let i = 0; i < words.length; i++) {
    const curWord = words[i];
    if(curWord === null) return res;
    if (curWord === "negative") {
      res = "-" + res;
    } else {
      const indexOfNums = nums.indexOf(curWord);
      if(indexOfNums !== -1) {
        res = res + indexOfNums;
      } else {
        const indexOfNums = tens.indexOf(curWord);
        if(indexOfNums !=== -1) {
          let teen = indexOfNums + 10;
          res += teen;
        } else {
          if (indexOfNums === 0) {
            res += "000";
          } else if(indexOfNums === 1) {
            res += "000000";
          } else if (indexOfNums === 2) {
            res += "000000000";
          }
        }
      }
    }
  }
  // console.log(Number.parseInt(res));
  return Number.parseInt(res);
};



