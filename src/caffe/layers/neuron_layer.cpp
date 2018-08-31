#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  skip_layer_ = (strstr(this->layer_param_.name().c_str(), "x2") != NULL
    || strstr(this->layer_param_.name().c_str(), "x1") != NULL
    || (strstr(this->layer_param_.name().c_str(), "blk") != NULL && strstr(this->layer_param_.name().c_str(), "5") == NULL)
    ) ? true : false;
  if (skip_layer_) {
    top[0]->ShareData(*bottom[0]);
    top[0]->ShareDiff(*bottom[0]);
  }
}

INSTANTIATE_CLASS(NeuronLayer);

}  // namespace caffe
