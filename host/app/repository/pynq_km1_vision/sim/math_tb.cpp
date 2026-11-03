#include "../src/hls/segment_math.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
int main(){
 unsigned count=0;
 for(unsigned bg: {0u,0xffffffu,0x2180f0u})
 for(unsigned t: {0u,1u,100u,764u,765u})
 for(unsigned p=0;p<0x1000000;p+=997){
   int d=std::abs(int(p&255)-int(bg&255))+std::abs(int((p>>8)&255)-int((bg>>8)&255))+std::abs(int((p>>16)&255)-int((bg>>16)&255));
   assert(segment_pixel(p,bg,t)==(d>int(t)?255u:0u));++count;
 }
 std::cout<<"passed "<<count<<" pixel cases (portable arithmetic only)\n";
}
