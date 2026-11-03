// Quadrature x4 decoder and atomic timestamp/count snapshot at FCLK 100 MHz.
// AXI GPIO ch1 -> cmd; ch2 <- data. cmd[8] toggle requests snapshot;
// cmd[3:0] selects read register. Same clock as AXI GPIO.
module conveyor_encoder(input wire clk, input wire resetn,
 input wire encoder_a, input wire encoder_b, input wire [31:0] cmd,
 output reg [31:0] data);
 (* ASYNC_REG = "TRUE" *) reg [1:0] a_sync,b_sync;
 reg [1:0] previous;
 reg signed [31:0] count;
 reg [63:0] ticks;
 reg [31:0] errors;
 reg ack;
 reg [31:0] snap_count,snap_errors;
 reg [63:0] snap_ticks;
 wire [1:0] ab={a_sync[1],b_sync[1]};
 always @(posedge clk) begin
  if (!resetn) begin
   a_sync<=0;b_sync<=0;previous<=0;count<=0;ticks<=0;errors<=0;
   ack<=0;snap_count<=0;snap_ticks<=0;snap_errors<=0;
  end else begin
   a_sync<={a_sync[0],encoder_a};b_sync<={b_sync[0],encoder_b};
   ticks<=ticks+1;previous<=ab;
   case ({previous,ab})
    4'b0001,4'b0111,4'b1110,4'b1000: count<=count+1;
    4'b0010,4'b1011,4'b1101,4'b0100: count<=count-1;
    default: if ((previous^ab)==2'b11) errors<=errors+1;
   endcase
   if (cmd[8]!=ack) begin
    snap_count<=count;snap_ticks<=ticks;snap_errors<=errors;ack<=cmd[8];
   end
  end
 end
 always @* begin
  case(cmd[3:0])
   0: data=32'h4b4d3501; // KM5 protocol 1
   1: data={31'b0,ack};
   2: data=snap_count;
   3: data=snap_ticks[31:0];
   4: data=snap_ticks[63:32];
   5: data=snap_errors;
   default: data=0;
  endcase
 end
endmodule
