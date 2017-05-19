var candy = function (ratings) {
    let res = 1;
    let totalc = ratings.length;
    let prevc = 1;
    for (let i = 1; i < totalc; i++) {
        if (ratings[i - 1] < ratings[i]) {
            prevc++;
            res += prevc;
        } else if (ratings[i - 1] == ratings[i]) {
            res++;
            prevc = 1;
        } else {
            let decStartIndex = i - 1;
            while (ratings[i - 1] > ratings[i] && i < totalc) {
                i++;
            }
            let c4start = Math.max(prevc, i - decStartIndex);  // A
            let cAfterStart = (i - decStartIndex) * (i - decStartIndex - 1) / 2;  // B
            res += c4start + cAfterStart - prevc;  // A + B - prev
            prevc = 1;
            i--; // move the pointer back to last decreasing rating
        }
    }
    return res;
}