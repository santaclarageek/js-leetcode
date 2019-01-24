//solution
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


var minCut = function(s) {
    for (let i=-1;i<s.length;i++) minCut[i] = i;
    for (let i=0;i<Math.floor(s.length+1/2);i++){
        for (let j=0;i-j>=0 && i+j<=s.length-1 && s[i-j]===s[i+j];j++)
            minCut[i+j] = Math.min(minCut[i+j], minCut[i-j-1]+1)
        for (let j=0;i-j-1>=0 && i+j<=s.length-1 && s[i-j-1]===s[i+j];j++)
            minCut[i+j] = Math.min(minCut[i+j], minCut[i-j-2]+1);
    }
    return minCut[s.length-1];
};