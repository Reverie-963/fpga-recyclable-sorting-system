#include <queue>
#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include "../src/hls/morph_stream.hpp"
struct Stream{
 std::queue<unsigned char> q;
 unsigned char read(){assert(!q.empty());auto v=q.front();q.pop();return v;}
 void write(unsigned char v){q.push(v);}
};
std::vector<unsigned char> reference(const std::vector<unsigned char>& a,int w,int h,bool erode){
 std::vector<unsigned char> b(w*h);
 for(int y=0;y<h;y++)for(int x=0;x<w;x++){
  unsigned char v=erode?255:0;
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){
   int xx=x+dx,yy=y+dy;unsigned char p=xx<0||xx>=w||yy<0||yy>=h?0:a[yy*w+xx];
   v=erode?(v&p):(v|p);
  }b[y*w+x]=v;
 }return b;
}
int main(){
 std::mt19937 rng(7);unsigned cases=0;
 for(int w: {1,2,3,7,32,640,1920})for(int h: {1,2,3,7,19})for(int mode=0;mode<4;mode++){
  std::vector<unsigned char>a(w*h);for(auto&v:a)v=mode==0?0:mode==1?255:(rng()%100<(mode==2?40:90)?255:0);
  Stream in,mid,out;for(auto v:a)in.write(v);
  morph3<true>(in,mid,w,h);morph3<false>(mid,out,w,h);
  auto expected=reference(reference(a,w,h,true),w,h,false);
  for(auto v:expected)assert(out.read()==v);
  assert(in.q.empty()&&mid.q.empty()&&out.q.empty());cases++;
 }
 std::cout<<"passed "<<cases<<" morphology frames incl. width=1920 and border cases\n";
}
