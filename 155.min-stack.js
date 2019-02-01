/*
 * @lc app=leetcode id=155 lang=javascript
 *
 * [155] Min Stack
 */
/**
 * initialize your data structure here.
 */

var MinStack = function() {
    //constructor是用于创建并赋值初始值的
      this.stack = [];
      this.min = [];
    };
    
    /** 
     * @param {number} x
     * @return {void}
     */
    MinStack.prototype.push = function(x) {
    //   this.stack.push(x);
      
    //   if (!this.min.length) {
    //     this.min.push(x);
    //   } else {
    //     const len = this.min.length;
    //     const last = this.min[len - 1];
    //     if (x < last) {
    //       this.min.push(x);
    //     } else {
    //       this.min.push(last);
    //     }
    //   }
        
        //或者这样的写法
        this.stack.push(x)
        let minS = this.min[this.min.length - 1]
        //console.log("1 : ",minS)
        this.min.length === 0 ? (minS = x) : "" 
        //"" 在这里只是占位符，，没有实质性用途
        //console.log("2 : ",minS)
        this.min.push(Math.min(x, minS))
    };
    
    /**
     * @return {void}
     */
    MinStack.prototype.pop = function() {
      this.stack.pop();
      this.min.pop();
    };
    
    /**
     * @return {number}
     */
    MinStack.prototype.top = function() {
      return this.stack[this.stack.length - 1];
    };
    
    /**
     * @return {number}
     */
    MinStack.prototype.getMin = function() {
      return this.min[this.min.length - 1];
    };

/** 
 * Your MinStack object will be instantiated and called as such:
 * var obj = new MinStack()
 * obj.push(x)
 * obj.pop()
 * var param_3 = obj.top()
 * var param_4 = obj.getMin()
 */

