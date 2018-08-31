#include <algorithm>
#include <vector>
#include <string>
#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"

namespace caffe {
// mean and var from x1bn to x1conv
extern void* bn_mean_temp;
extern void* bn_var_temp;
// forwarding mean and var from x1conv to x2conv
const void* temp_mean_blob;
const void* temp_var_blob;
bool is_allocated_global = false;
void *d_beta_chw_extern;
void *d_gamma_chw_extern;
void *A_extern;
void *B_extern;
void *C_extern;
void *forward_temp;

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (!is_allocated_global) {
    cudaMalloc(&d_gamma_chw_extern, 831744 * sizeof(double));
    cudaMalloc(&d_beta_chw_extern, 831744 * sizeof(Dtype));
    cudaMalloc(&A_extern, 831744 * sizeof(Dtype));
    cudaMalloc(&B_extern, 831744 * sizeof(Dtype));
    cudaMalloc(&C_extern, 831744 * sizeof(Dtype));
    cudaMalloc(&forward_temp, 831744 * sizeof(Dtype));

    is_allocated_global = true;
  }
  d_gamma_chw = static_cast<Dtype*> (d_gamma_chw_extern);
  d_beta_chw = static_cast<Dtype*> (d_beta_chw_extern);
  A = static_cast<Dtype*> (A_extern);
  B = static_cast<Dtype*> (B_extern);
  C = static_cast<Dtype*> (C_extern);

  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  mean_var_fusion_ = (strstr(this->layer_param_.name().c_str(), "x1") != NULL) ? true : false;
  norm_fusion_ = (strstr(this->layer_param_.name().c_str(), "x2") != NULL ||
      strstr(this->layer_param_.name().c_str(), "x1") != NULL ||
      (strstr(this->layer_param_.name().c_str(), "blk") != NULL && strstr(this->layer_param_.name().c_str(), "5") == NULL)
      )? true : false;
  is_blk_ = (strstr(this->layer_param_.name().c_str(), "blk") != NULL && strstr(this->layer_param_.name().c_str(), "5") == NULL)? true : false;
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
        << " vs. bottom[" << bottom_id << "]: "
        << bottom[bottom_id]->shape_string();
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  if (norm_fusion_) {
    vector<int> relu;
    relu.push_back(num_);
    relu.push_back(conv_in_channels_);
    relu.push_back(conv_input_shape_.cpu_data()[1]);
    relu.push_back(conv_input_shape_.cpu_data()[2]);
    relu_inp_.Reshape(relu);

    vector<int> spatial_multiplier_shape(1, out_spatial_dim_);
    spatial_multiplier_.Reshape(spatial_multiplier_shape);
    caffe_set(spatial_multiplier_.count(), Dtype(1),
        spatial_multiplier_.mutable_cpu_data());
  }

  // usually x1
  if (norm_fusion_ && mean_var_fusion_) {
    vector<int> ch;
    ch.push_back(conv_in_channels_);
    gamma_.Reshape(ch);
    beta_.Reshape(ch);
    mean_.Reshape(ch);
    var_.Reshape(ch);

    // vars to pass
    vector<int> ch_o;
    ch_o.push_back(conv_out_channels_);
    mean_to_pass_.Reshape(ch_o);
    var_to_pass_.Reshape(ch_o);

    // temporal storages for reduction
    vector<int> ch_bx_shape;
    // bx * channels
    ch_bx_shape.push_back((conv_out_spatial_dim_ + 127) / 128);
    ch_bx_shape.push_back(conv_out_channels_);
    // reshape the twos
    x_accum_temp_.Reshape(ch_bx_shape);
    x2_accum_temp_.Reshape(ch_bx_shape);

    // init gamma
    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(1);
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(&this->gamma_);

    FillerParameter bias_param;
    bias_param.set_type("constant");
    bias_param.set_value(0);
    shared_ptr<Filler<Dtype> > filler_bias(GetFiller<Dtype>(bias_param));
    filler_bias->Fill(&this->beta_);

    temp_mean_blob = &(mean_to_pass_);
    temp_var_blob = &(var_to_pass_);
    // from bn
    mean_.set_gpu_data(static_cast<Dtype *>(bn_mean_temp));
    var_.set_gpu_data(static_cast<Dtype*>(bn_var_temp));
  }

  // x2 conv
  else if (!mean_var_fusion_ && norm_fusion_) {
    vector<int> ch;
    ch.push_back(conv_in_channels_);
    gamma_.Reshape(ch);
    beta_.Reshape(ch);
    mean_.Reshape(ch);
    var_.Reshape(ch);

    // init gamma
    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(1);
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(&this->gamma_);

    // init beta
    FillerParameter bias_param;
    bias_param.set_type("constant");
    bias_param.set_value(0);
    shared_ptr<Filler<Dtype> > filler_bias(GetFiller<Dtype>(bias_param));
    filler_bias->Fill(&this->beta_);

    if (is_blk_) {
      // from bn
      mean_.set_gpu_data(static_cast<Dtype *>(bn_mean_temp));
      var_.set_gpu_data(static_cast<Dtype*>(bn_var_temp));
    }
    else if (!is_blk_) {
      mean_.ShareData(*(static_cast<const Blob<Dtype>* >(temp_mean_blob)));
      var_.ShareData(*(static_cast<const Blob<Dtype>* >(temp_var_blob)));
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm_mean_var_fusion(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  Dtype* x_accum_temp = x_accum_temp_.mutable_gpu_data();
  Dtype* x2_accum_temp = x2_accum_temp_.mutable_gpu_data();;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm_mean_var_fusion<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g, x_accum_temp, x2_accum_temp);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm_mean_var_norm_fusion(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col, int n) {
  Dtype* col_buff = const_cast<Dtype*> (input);
  Dtype* x_accum_temp = x_accum_temp_.mutable_gpu_data();
  Dtype* x2_accum_temp = x2_accum_temp_.mutable_gpu_data();;
  Dtype* temp_ = static_cast<Dtype*> (forward_temp);
  if (!is_1x1_) {
    LOG(FATAL) << "does not reach here";
    if (!skip_im2col) {
      // does not reach here
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.mutable_gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    norm_and_save_relu(kernel_dim_, conv_out_spatial_dim_, col_buff + col_offset_ * g, mean_.gpu_data(), var_.gpu_data(), gamma_.gpu_data(), beta_.gpu_data(), relu_inp_.mutable_gpu_data() + n * bottom_dim_, temp_);
    caffe_gpu_gemm_mean_var_fusion<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, temp_ + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g, x_accum_temp, x2_accum_temp);
  }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm_norm_fusion(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col, int n) {
  Dtype* col_buff = const_cast<Dtype*> (input);
  Dtype* temp_ = static_cast<Dtype*> (forward_temp);
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu_norm(input, col_buffer_.mutable_gpu_data(), n);
    }
    col_buff = col_buffer_.mutable_gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    if (!is_1x1_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
          group_, conv_out_spatial_dim_, kernel_dim_,
          (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)0., output + output_offset_ * g);
    }
    else {
      norm_and_save_relu(kernel_dim_, conv_out_spatial_dim_, col_buff + col_offset_ * g, mean_.gpu_data(), var_.gpu_data(), gamma_.gpu_data(), beta_.gpu_data(), relu_inp_.mutable_gpu_data() + n * bottom_dim_, temp_);
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
          group_, conv_out_spatial_dim_, kernel_dim_,
          (Dtype)1., weights + weight_offset_ * g, temp_ + col_offset_ * g,
          (Dtype)0., output + output_offset_ * g);
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm_gather(const Dtype* output,
    const Dtype* weights, const Dtype* x_relu, const Dtype* x_bn, Dtype* input, int n) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    if (!is_1x1_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
          conv_out_spatial_dim_, conv_out_channels_ / group_,
          (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
          (Dtype)0., col_buff + col_offset_ * g);
    }
    else { // 1x1 case
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
          conv_out_spatial_dim_, conv_out_channels_ / group_,
          (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
          (Dtype)0., col_buff + col_offset_ * g);
      relu_and_gather(x_relu, x_bn, input, n);
    }
  }
  if (!is_1x1_) {
    conv_col2im_gpu_gather(col_buff, input, x_relu, x_bn);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm_relu(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu_relu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    if (!is_1x1_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
          kernel_dim_, conv_out_spatial_dim_,
          (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)1., weights + weight_offset_ * g);
    }
    else {
      ReLU_inplace(kernel_dim_ * conv_out_spatial_dim_, const_cast<Dtype*> (col_buff));
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
          kernel_dim_, conv_out_spatial_dim_,
          (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
          (Dtype)1., weights + weight_offset_ * g);
    }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_reduction(Dtype* output,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, conv_in_channels_, conv_input_shape_.cpu_data()[1] * conv_input_shape_.cpu_data()[2], 1.,
      input, spatial_multiplier_.gpu_data(), 0., output);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
