//#pragma optionNV unroll all
uniform float weightVector[N_WEIGHTS];
uniform vec3 camPos;
varying vec3 posInWorldSpace;
varying vec3 normal;

void main(){
  //gl_FragColor = vec4(posInWorldSpace.y < 400.0 ? posInWorldSpace.y/400.0:0,
  //                    weightVector[0], 0, 1);

  vec3 viewDirection = normalize(posInWorldSpace - camPos);
  
  // input is {X,Normal,View,Lightpos}
  float x[12] = float[](posInWorldSpace.x/600., posInWorldSpace.y/600., posInWorldSpace.z/600.,
			normal.x, normal.y, normal.z,
			viewDirection.x, viewDirection.y, viewDirection.z,
			0.5,0.1,0.1);
  vec3 c;
  
  int W1p = 0;
  int b1p = 12 * N_LAYER1;
  int W2p = b1p + N_LAYER1;
  int b2p = W2p + N_LAYER1*N_LAYER2;
  int W3p = b2p + N_LAYER2;
  int b3p = W3p + N_LAYER2*N_LAYER3;

  float z1[N_LAYER1];
  float z2[N_LAYER2];
  float z3[N_LAYER3];
  int i,j;
  for (i=0;i<N_LAYER1;i++){
    float s =0;
    for (j=0;j<12;j++){
      s += x[j]*weightVector[W1p+N_LAYER1*j+i];
    }
    s += weightVector[b1p+i];
    z1[i] = s < 0 ? 0 : s;
  }
  
  for (i=0;i<N_LAYER2;i++){
    float s =0;
    for (j=0;j<N_LAYER1;j++){
      s += z1[j]*weightVector[W2p+N_LAYER2*j+i];
    }
    s += weightVector[b2p+i];
    z2[i] = s < 0 ? 0 : s;
  }

  for (i=0;i<N_LAYER3;i++){
    float s = 0;
    for (j=0;j<N_LAYER2;j++){
      s += z2[j]*weightVector[W3p+N_LAYER3*j+i];
    }
    s += weightVector[b3p+i];
    z3[i] = 1 / (1 + exp(-s));
  }
  gl_FragColor = vec4(z3[0], z3[1], z3[2], 1);
}
