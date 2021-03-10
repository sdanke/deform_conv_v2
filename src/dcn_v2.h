#include <torch/script.h>

void init();

at::Tensor
deform_conv_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const int64_t kernel_h,
               const int64_t kernel_w,
               const int64_t stride_h,
               const int64_t stride_w,
               const int64_t pad_h,
               const int64_t pad_w,
               const int64_t dilation_h,
               const int64_t dilation_w,
               const int64_t group,
               const int64_t deformable_group,
               const int64_t im2col_step);


std::vector<at::Tensor>
deform_conv_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &grad_output,
                const int64_t kernel_h, 
                const int64_t kernel_w,
                const int64_t stride_h, 
                const int64_t stride_w,
                const int64_t pad_h, 
                const int64_t pad_w,
                const int64_t dilation_h, 
                const int64_t dilation_w,
                const int64_t group,
                const int64_t deformable_group,
                const int64_t im2col_step);

std::vector<at::Tensor>
deform_conv_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &grad_output,
                const int64_t kernel_h, 
                const int64_t kernel_w,
                const int64_t stride_h, 
                const int64_t stride_w,
                const int64_t pad_h, 
                const int64_t pad_w,
                const int64_t dilation_h, 
                const int64_t dilation_w,
                const int64_t group,
                const int64_t deformable_group,
                const int64_t im2col_step);

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_forward(const at::Tensor &input,
                             const at::Tensor &bbox,
                             const at::Tensor &trans,
                             const int64_t no_trans,
                             const double spatial_scale,
                             const int64_t output_dim,
                             const int64_t group_size,
                             const int64_t pooled_size,
                             const int64_t part_size,
                             const int64_t sample_per_part,
                             const double trans_std);

std::tuple<at::Tensor, at::Tensor>
deform_psroi_pooling_backward(const at::Tensor &out_grad,
                              const at::Tensor &input,
                              const at::Tensor &bbox,
                              const at::Tensor &trans,
                              const at::Tensor &top_count,
                              const int64_t no_trans,
                              const double spatial_scale,
                              const int64_t output_dim,
                              const int64_t group_size,
                              const int64_t pooled_size,
                              const int64_t part_size,
                              const int64_t sample_per_part,
                              const double trans_std);

at::Tensor
modulated_deform_conv_forward(const at::Tensor &input,
               const at::Tensor &weight,
               const at::Tensor &bias,
               const at::Tensor &offset,
               const at::Tensor &mask,
               const int64_t kernel_h,
               const int64_t kernel_w,
               const int64_t stride_h,
               const int64_t stride_w,
               const int64_t pad_h,
               const int64_t pad_w,
               const int64_t dilation_h,
               const int64_t dilation_w,
               const int64_t group, 
               const int64_t deformable_group,
               const int64_t im2col_step);

std::vector<at::Tensor>
modulated_deform_conv_backward(const at::Tensor &input,
                const at::Tensor &weight,
                const at::Tensor &bias,
                const at::Tensor &offset,
                const at::Tensor &mask,
                const at::Tensor &grad_output,
                const int64_t kernel_h, 
                const int64_t kernel_w,
                const int64_t stride_h, 
                const int64_t stride_w,
                const int64_t pad_h, 
                const int64_t pad_w,
                const int64_t dilation_h, 
                const int64_t dilation_w,
                const int64_t group,
                const int64_t deformable_group,
                const int64_t im2col_step)
