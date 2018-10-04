	// longest consecutive


process.stdin.resume();
process.stdin.setEncoding('utf8');
const fs = require('fs');
var stdin = '';

process.stdin.on('data', function (chunk) {
  stdin += chunk;
  
}).on('end', function() {
  var lines = stdin.trim().split('\n');
  
  for(var i=0; i<lines.length; i++) {
    process.stdout.write(longestConsecutive(parseInt(lines[i])).toString());
  }
});


	var longestConsecutive = function(nums) {


    if (nums === null || nums.length === 0) {

        return 0;
    }
            while (set.has(n)) {
                set.delete(n);
                n = n + 1;
                temp++;
            }
            length = Math.max(temp, length);
        }
    }
    return length;
};
