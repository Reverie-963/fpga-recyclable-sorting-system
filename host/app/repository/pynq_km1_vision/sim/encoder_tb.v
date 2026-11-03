`timescale 1ns/1ps
module encoder_tb;
 reg clk=0,resetn=0,a=0,b=0;reg [31:0] cmd=0;wire[31:0]data;
 conveyor_encoder dut(clk,resetn,a,b,cmd,data);
 always #5 clk=~clk;
 task state(input aa,input bb);begin @(negedge clk);a=aa;b=bb;repeat(6) @(negedge clk);end endtask
 task snap;begin @(negedge clk);cmd=(cmd^256)&256;repeat(3) @(negedge clk);end endtask
 task expect_reg(input [3:0] addr,input[31:0] expected);begin
  cmd=(cmd&256)|addr;#1;if(data!==expected)begin $display("FAIL addr %d got %h expected %h",addr,data,expected);$fatal;end
 end endtask
 reg [31:0] oldtime;
 initial begin
  repeat(3) @(negedge clk);resetn=1;
  state(0,1);state(1,1);state(1,0);state(0,0);
  snap;expect_reg(0,32'h4b4d3501);expect_reg(1,1);expect_reg(2,4);expect_reg(5,0);
  cmd=(cmd&256)|3;#1;oldtime=data;
  state(1,0);state(1,1);state(0,1);state(0,0);
  expect_reg(2,4); // latched sample stays atomic while live count changes
  snap;expect_reg(2,0);expect_reg(1,0);cmd=(cmd&256)|3;#1;
  if(data<=oldtime)$fatal;
  state(1,1);snap;expect_reg(5,1);expect_reg(2,0);
  @(negedge clk);resetn=0;repeat(3)@(negedge clk);resetn=1;state(0,0);snap;
  $display("PASS quadrature forward reverse atomic snapshot timestamp illegal transition reset");$finish;
 end
endmodule
