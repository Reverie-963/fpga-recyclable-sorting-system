#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <cassert>
#include <vector>
#include <cstdlib>
typedef ap_axiu<32,0,0,0> axis_word;
void segment_open(hls::stream<axis_word>&,hls::stream<axis_word>&,unsigned,unsigned,unsigned,unsigned);
static std::vector<unsigned char> ref(const std::vector<unsigned char>& a,int w,int h,bool erode){
 std::vector<unsigned char>b(w*h);
 for(int y=0;y<h;y++)for(int x=0;x<w;x++){
  unsigned char v=erode?255:0;
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int xx=x+dx,yy=y+dy;unsigned char p=xx<0||xx>=w||yy<0||yy>=h?0:a[yy*w+xx];v=erode?(v&p):(v|p);
  }b[y*w+x]=v;
 }return b;
}
int main(){
 for(unsigned w:{1u,17u,640u}){
  unsigned h=19,n=w*h;hls::stream<axis_word> input,output;std::vector<unsigned char> mask;
  for(unsigned i=0;i<n;i++){
   unsigned p=(i*7193)&0xffffff;axis_word v;v.data=p;v.keep=15;v.strb=15;v.last=i==n-1;input.write(v);
   int d=std::abs(int(p&255)-240)+std::abs(int((p>>8)&255)-240)+std::abs(int((p>>16)&255)-240);mask.push_back(d>100?255:0);
  }
  segment_open(input,output,w,h,0xf0f0f0,100);auto expected=ref(ref(mask,w,h,true),w,h,false);
  for(unsigned i=0;i<n;i++){auto v=output.read();assert(v.data==expected[i]);assert(v.keep==15);assert(v.last==(i==n-1));}
  assert(input.empty()&&output.empty());
 }
}
