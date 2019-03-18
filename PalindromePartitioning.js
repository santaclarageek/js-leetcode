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