��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
d
count_21VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_21
]
count_21/Read/ReadVariableOpReadVariableOpcount_21*
_output_shapes
: *
dtype0
d
total_21VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_21
]
total_21/Read/ReadVariableOpReadVariableOptotal_21*
_output_shapes
: *
dtype0
�
training_42/SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_42/SGD/momentum
}
,training_42/SGD/momentum/Read/ReadVariableOpReadVariableOptraining_42/SGD/momentum*
_output_shapes
: *
dtype0
�
training_42/SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_42/SGD/learning_rate
�
1training_42/SGD/learning_rate/Read/ReadVariableOpReadVariableOptraining_42/SGD/learning_rate*
_output_shapes
: *
dtype0
~
training_42/SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_42/SGD/decay
w
)training_42/SGD/decay/Read/ReadVariableOpReadVariableOptraining_42/SGD/decay*
_output_shapes
: *
dtype0
|
training_42/SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_42/SGD/iter
u
(training_42/SGD/iter/Read/ReadVariableOpReadVariableOptraining_42/SGD/iter*
_output_shapes
: *
dtype0	
r
dense_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_65/bias
k
!dense_65/bias/Read/ReadVariableOpReadVariableOpdense_65/bias*
_output_shapes
:*
dtype0
z
dense_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_65/kernel
s
#dense_65/kernel/Read/ReadVariableOpReadVariableOpdense_65/kernel*
_output_shapes

:
*
dtype0
r
dense_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_64/bias
k
!dense_64/bias/Read/ReadVariableOpReadVariableOpdense_64/bias*
_output_shapes
:
*
dtype0
z
dense_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_64/kernel
s
#dense_64/kernel/Read/ReadVariableOpReadVariableOpdense_64/kernel*
_output_shapes

:

*
dtype0
r
dense_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_63/bias
k
!dense_63/bias/Read/ReadVariableOpReadVariableOpdense_63/bias*
_output_shapes
:
*
dtype0
z
dense_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_63/kernel
s
#dense_63/kernel/Read/ReadVariableOpReadVariableOpdense_63/kernel*
_output_shapes

:

*
dtype0
{
serving_default_input_22Placeholder*'
_output_shapes
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_22dense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_16256

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
.
0
1
2
3
$4
%5*
.
0
1
2
3
$4
%5*
* 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
* 
:
3iter
	4decay
5learning_rate
6momentum*

7serving_default* 

0
1*

0
1*
* 
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

=trace_0* 

>trace_0* 
_Y
VARIABLE_VALUEdense_63/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_63/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Dtrace_0* 

Etrace_0* 
_Y
VARIABLE_VALUEdense_64/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_64/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
_Y
VARIABLE_VALUEdense_65/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_65/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

M0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
WQ
VARIABLE_VALUEtraining_42/SGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining_42/SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEtraining_42/SGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtraining_42/SGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
H
N	variables
O	keras_api
	Ptotal
	Qcount
R
_fn_kwargs*

P0
Q1*

N	variables*
VP
VARIABLE_VALUEtotal_214keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_214keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_63/kernel/Read/ReadVariableOp!dense_63/bias/Read/ReadVariableOp#dense_64/kernel/Read/ReadVariableOp!dense_64/bias/Read/ReadVariableOp#dense_65/kernel/Read/ReadVariableOp!dense_65/bias/Read/ReadVariableOp(training_42/SGD/iter/Read/ReadVariableOp)training_42/SGD/decay/Read/ReadVariableOp1training_42/SGD/learning_rate/Read/ReadVariableOp,training_42/SGD/momentum/Read/ReadVariableOptotal_21/Read/ReadVariableOpcount_21/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_16441
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_63/kerneldense_63/biasdense_64/kerneldense_64/biasdense_65/kerneldense_65/biastraining_42/SGD/itertraining_42/SGD/decaytraining_42/SGD/learning_ratetraining_42/SGD/momentumtotal_21count_21*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_16487��
�"
�
__inference__traced_save_16441
file_prefix.
*savev2_dense_63_kernel_read_readvariableop,
(savev2_dense_63_bias_read_readvariableop.
*savev2_dense_64_kernel_read_readvariableop,
(savev2_dense_64_bias_read_readvariableop.
*savev2_dense_65_kernel_read_readvariableop,
(savev2_dense_65_bias_read_readvariableop3
/savev2_training_42_sgd_iter_read_readvariableop	4
0savev2_training_42_sgd_decay_read_readvariableop<
8savev2_training_42_sgd_learning_rate_read_readvariableop7
3savev2_training_42_sgd_momentum_read_readvariableop'
#savev2_total_21_read_readvariableop'
#savev2_count_21_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_63_kernel_read_readvariableop(savev2_dense_63_bias_read_readvariableop*savev2_dense_64_kernel_read_readvariableop(savev2_dense_64_bias_read_readvariableop*savev2_dense_65_kernel_read_readvariableop(savev2_dense_65_bias_read_readvariableop/savev2_training_42_sgd_iter_read_readvariableop0savev2_training_42_sgd_decay_read_readvariableop8savev2_training_42_sgd_learning_rate_read_readvariableop3savev2_training_42_sgd_momentum_read_readvariableop#savev2_total_21_read_readvariableop#savev2_count_21_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*S
_input_shapesB
@: :

:
:

:
:
:: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
� 
�
 __inference__wrapped_model_16026
input_22I
7model_21_dense_63_matmul_readvariableop_dense_63_kernel:

D
6model_21_dense_63_biasadd_readvariableop_dense_63_bias:
I
7model_21_dense_64_matmul_readvariableop_dense_64_kernel:

D
6model_21_dense_64_biasadd_readvariableop_dense_64_bias:
I
7model_21_dense_65_matmul_readvariableop_dense_65_kernel:
D
6model_21_dense_65_biasadd_readvariableop_dense_65_bias:
identity��(model_21/dense_63/BiasAdd/ReadVariableOp�'model_21/dense_63/MatMul/ReadVariableOp�(model_21/dense_64/BiasAdd/ReadVariableOp�'model_21/dense_64/MatMul/ReadVariableOp�(model_21/dense_65/BiasAdd/ReadVariableOp�'model_21/dense_65/MatMul/ReadVariableOp�
'model_21/dense_63/MatMul/ReadVariableOpReadVariableOp7model_21_dense_63_matmul_readvariableop_dense_63_kernel*
_output_shapes

:

*
dtype0�
model_21/dense_63/MatMulMatMulinput_22/model_21/dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
(model_21/dense_63/BiasAdd/ReadVariableOpReadVariableOp6model_21_dense_63_biasadd_readvariableop_dense_63_bias*
_output_shapes
:
*
dtype0�
model_21/dense_63/BiasAddBiasAdd"model_21/dense_63/MatMul:product:00model_21/dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
model_21/dense_63/ReluRelu"model_21/dense_63/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
'model_21/dense_64/MatMul/ReadVariableOpReadVariableOp7model_21_dense_64_matmul_readvariableop_dense_64_kernel*
_output_shapes

:

*
dtype0�
model_21/dense_64/MatMulMatMul$model_21/dense_63/Relu:activations:0/model_21/dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
(model_21/dense_64/BiasAdd/ReadVariableOpReadVariableOp6model_21_dense_64_biasadd_readvariableop_dense_64_bias*
_output_shapes
:
*
dtype0�
model_21/dense_64/BiasAddBiasAdd"model_21/dense_64/MatMul:product:00model_21/dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
model_21/dense_64/ReluRelu"model_21/dense_64/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
'model_21/dense_65/MatMul/ReadVariableOpReadVariableOp7model_21_dense_65_matmul_readvariableop_dense_65_kernel*
_output_shapes

:
*
dtype0�
model_21/dense_65/MatMulMatMul$model_21/dense_64/Relu:activations:0/model_21/dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(model_21/dense_65/BiasAdd/ReadVariableOpReadVariableOp6model_21_dense_65_biasadd_readvariableop_dense_65_bias*
_output_shapes
:*
dtype0�
model_21/dense_65/BiasAddBiasAdd"model_21/dense_65/MatMul:product:00model_21/dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
model_21/dense_65/SoftmaxSoftmax"model_21/dense_65/BiasAdd:output:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#model_21/dense_65/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^model_21/dense_63/BiasAdd/ReadVariableOp(^model_21/dense_63/MatMul/ReadVariableOp)^model_21/dense_64/BiasAdd/ReadVariableOp(^model_21/dense_64/MatMul/ReadVariableOp)^model_21/dense_65/BiasAdd/ReadVariableOp(^model_21/dense_65/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2T
(model_21/dense_63/BiasAdd/ReadVariableOp(model_21/dense_63/BiasAdd/ReadVariableOp2R
'model_21/dense_63/MatMul/ReadVariableOp'model_21/dense_63/MatMul/ReadVariableOp2T
(model_21/dense_64/BiasAdd/ReadVariableOp(model_21/dense_64/BiasAdd/ReadVariableOp2R
'model_21/dense_64/MatMul/ReadVariableOp'model_21/dense_64/MatMul/ReadVariableOp2T
(model_21/dense_65/BiasAdd/ReadVariableOp(model_21/dense_65/BiasAdd/ReadVariableOp2R
'model_21/dense_65/MatMul/ReadVariableOp'model_21/dense_65/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_22
�	
�
(__inference_model_21_layer_call_fn_16217
input_22!
dense_63_kernel:


dense_63_bias:
!
dense_64_kernel:


dense_64_bias:
!
dense_65_kernel:

dense_65_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_22dense_63_kerneldense_63_biasdense_64_kerneldense_64_biasdense_65_kerneldense_65_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_21_layer_call_and_return_conditional_losses_16171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_22
�
�
C__inference_model_21_layer_call_and_return_conditional_losses_16171

inputs*
dense_63_dense_63_kernel:

$
dense_63_dense_63_bias:
*
dense_64_dense_64_kernel:

$
dense_64_dense_64_bias:
*
dense_65_dense_65_kernel:
$
dense_65_dense_65_bias:
identity�� dense_63/StatefulPartitionedCall� dense_64/StatefulPartitionedCall� dense_65/StatefulPartitionedCall�
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinputsdense_63_dense_63_kerneldense_63_dense_63_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_16044�
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_dense_64_kerneldense_64_dense_64_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_16059�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_dense_65_kerneldense_65_dense_65_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_16074x
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
(__inference_model_21_layer_call_fn_16088
input_22!
dense_63_kernel:


dense_63_bias:
!
dense_64_kernel:


dense_64_bias:
!
dense_65_kernel:

dense_65_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_22dense_63_kerneldense_63_biasdense_64_kerneldense_64_biasdense_65_kerneldense_65_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_21_layer_call_and_return_conditional_losses_16079o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_22
�
�
C__inference_model_21_layer_call_and_return_conditional_losses_16243
input_22*
dense_63_dense_63_kernel:

$
dense_63_dense_63_bias:
*
dense_64_dense_64_kernel:

$
dense_64_dense_64_bias:
*
dense_65_dense_65_kernel:
$
dense_65_dense_65_bias:
identity�� dense_63/StatefulPartitionedCall� dense_64/StatefulPartitionedCall� dense_65/StatefulPartitionedCall�
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinput_22dense_63_dense_63_kerneldense_63_dense_63_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_16044�
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_dense_64_kerneldense_64_dense_64_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_16059�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_dense_65_kerneldense_65_dense_65_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_16074x
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_22
�

�
C__inference_dense_63_layer_call_and_return_conditional_losses_16346

inputs7
%matmul_readvariableop_dense_63_kernel:

2
$biasadd_readvariableop_dense_63_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_63_kernel*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_63_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_dense_64_layer_call_fn_16353

inputs!
dense_64_kernel:


dense_64_bias:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_64_kerneldense_64_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_16059o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_65_layer_call_and_return_conditional_losses_16382

inputs7
%matmul_readvariableop_dense_65_kernel:
2
$biasadd_readvariableop_dense_65_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_65_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_65_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_65_layer_call_and_return_conditional_losses_16074

inputs7
%matmul_readvariableop_dense_65_kernel:
2
$biasadd_readvariableop_dense_65_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_65_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_65_bias*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_model_21_layer_call_fn_16267

inputs!
dense_63_kernel:


dense_63_bias:
!
dense_64_kernel:


dense_64_bias:
!
dense_65_kernel:

dense_65_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_63_kerneldense_63_biasdense_64_kerneldense_64_biasdense_65_kerneldense_65_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_21_layer_call_and_return_conditional_losses_16079o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
C__inference_model_21_layer_call_and_return_conditional_losses_16079

inputs*
dense_63_dense_63_kernel:

$
dense_63_dense_63_bias:
*
dense_64_dense_64_kernel:

$
dense_64_dense_64_bias:
*
dense_65_dense_65_kernel:
$
dense_65_dense_65_bias:
identity�� dense_63/StatefulPartitionedCall� dense_64/StatefulPartitionedCall� dense_65/StatefulPartitionedCall�
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinputsdense_63_dense_63_kerneldense_63_dense_63_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_16044�
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_dense_64_kerneldense_64_dense_64_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_16059�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_dense_65_kerneldense_65_dense_65_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_16074x
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_64_layer_call_and_return_conditional_losses_16364

inputs7
%matmul_readvariableop_dense_64_kernel:

2
$biasadd_readvariableop_dense_64_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_64_kernel*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_64_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
C__inference_model_21_layer_call_and_return_conditional_losses_16230
input_22*
dense_63_dense_63_kernel:

$
dense_63_dense_63_bias:
*
dense_64_dense_64_kernel:

$
dense_64_dense_64_bias:
*
dense_65_dense_65_kernel:
$
dense_65_dense_65_bias:
identity�� dense_63/StatefulPartitionedCall� dense_64/StatefulPartitionedCall� dense_65/StatefulPartitionedCall�
 dense_63/StatefulPartitionedCallStatefulPartitionedCallinput_22dense_63_dense_63_kerneldense_63_dense_63_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_16044�
 dense_64/StatefulPartitionedCallStatefulPartitionedCall)dense_63/StatefulPartitionedCall:output:0dense_64_dense_64_kerneldense_64_dense_64_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_64_layer_call_and_return_conditional_losses_16059�
 dense_65/StatefulPartitionedCallStatefulPartitionedCall)dense_64/StatefulPartitionedCall:output:0dense_65_dense_65_kerneldense_65_dense_65_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_16074x
IdentityIdentity)dense_65/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_63/StatefulPartitionedCall!^dense_64/StatefulPartitionedCall!^dense_65/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2D
 dense_63/StatefulPartitionedCall dense_63/StatefulPartitionedCall2D
 dense_64/StatefulPartitionedCall dense_64/StatefulPartitionedCall2D
 dense_65/StatefulPartitionedCall dense_65/StatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_22
�
�
(__inference_dense_65_layer_call_fn_16371

inputs!
dense_65_kernel:

dense_65_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_65_kerneldense_65_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_65_layer_call_and_return_conditional_losses_16074o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_64_layer_call_and_return_conditional_losses_16059

inputs7
%matmul_readvariableop_dense_64_kernel:

2
$biasadd_readvariableop_dense_64_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_64_kernel*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_64_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
C__inference_model_21_layer_call_and_return_conditional_losses_16303

inputs@
.dense_63_matmul_readvariableop_dense_63_kernel:

;
-dense_63_biasadd_readvariableop_dense_63_bias:
@
.dense_64_matmul_readvariableop_dense_64_kernel:

;
-dense_64_biasadd_readvariableop_dense_64_bias:
@
.dense_65_matmul_readvariableop_dense_65_kernel:
;
-dense_65_biasadd_readvariableop_dense_65_bias:
identity��dense_63/BiasAdd/ReadVariableOp�dense_63/MatMul/ReadVariableOp�dense_64/BiasAdd/ReadVariableOp�dense_64/MatMul/ReadVariableOp�dense_65/BiasAdd/ReadVariableOp�dense_65/MatMul/ReadVariableOp�
dense_63/MatMul/ReadVariableOpReadVariableOp.dense_63_matmul_readvariableop_dense_63_kernel*
_output_shapes

:

*
dtype0{
dense_63/MatMulMatMulinputs&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_63/BiasAdd/ReadVariableOpReadVariableOp-dense_63_biasadd_readvariableop_dense_63_bias*
_output_shapes
:
*
dtype0�
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_64/MatMul/ReadVariableOpReadVariableOp.dense_64_matmul_readvariableop_dense_64_kernel*
_output_shapes

:

*
dtype0�
dense_64/MatMulMatMuldense_63/Relu:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_64/BiasAdd/ReadVariableOpReadVariableOp-dense_64_biasadd_readvariableop_dense_64_bias*
_output_shapes
:
*
dtype0�
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_65/MatMul/ReadVariableOpReadVariableOp.dense_65_matmul_readvariableop_dense_65_kernel*
_output_shapes

:
*
dtype0�
dense_65/MatMulMatMuldense_64/Relu:activations:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_65/BiasAdd/ReadVariableOpReadVariableOp-dense_65_biasadd_readvariableop_dense_65_bias*
_output_shapes
:*
dtype0�
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_65/SoftmaxSoftmaxdense_65/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_65/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_dense_63_layer_call_fn_16335

inputs!
dense_63_kernel:


dense_63_bias:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_63_kerneldense_63_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_63_layer_call_and_return_conditional_losses_16044o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0**
_input_shapes
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_63_layer_call_and_return_conditional_losses_16044

inputs7
%matmul_readvariableop_dense_63_kernel:

2
$biasadd_readvariableop_dense_63_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_63_kernel*
_output_shapes

:

*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_63_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������
a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
C__inference_model_21_layer_call_and_return_conditional_losses_16328

inputs@
.dense_63_matmul_readvariableop_dense_63_kernel:

;
-dense_63_biasadd_readvariableop_dense_63_bias:
@
.dense_64_matmul_readvariableop_dense_64_kernel:

;
-dense_64_biasadd_readvariableop_dense_64_bias:
@
.dense_65_matmul_readvariableop_dense_65_kernel:
;
-dense_65_biasadd_readvariableop_dense_65_bias:
identity��dense_63/BiasAdd/ReadVariableOp�dense_63/MatMul/ReadVariableOp�dense_64/BiasAdd/ReadVariableOp�dense_64/MatMul/ReadVariableOp�dense_65/BiasAdd/ReadVariableOp�dense_65/MatMul/ReadVariableOp�
dense_63/MatMul/ReadVariableOpReadVariableOp.dense_63_matmul_readvariableop_dense_63_kernel*
_output_shapes

:

*
dtype0{
dense_63/MatMulMatMulinputs&dense_63/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_63/BiasAdd/ReadVariableOpReadVariableOp-dense_63_biasadd_readvariableop_dense_63_bias*
_output_shapes
:
*
dtype0�
dense_63/BiasAddBiasAdddense_63/MatMul:product:0'dense_63/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_63/ReluReludense_63/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_64/MatMul/ReadVariableOpReadVariableOp.dense_64_matmul_readvariableop_dense_64_kernel*
_output_shapes

:

*
dtype0�
dense_64/MatMulMatMuldense_63/Relu:activations:0&dense_64/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_64/BiasAdd/ReadVariableOpReadVariableOp-dense_64_biasadd_readvariableop_dense_64_bias*
_output_shapes
:
*
dtype0�
dense_64/BiasAddBiasAdddense_64/MatMul:product:0'dense_64/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_64/ReluReludense_64/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_65/MatMul/ReadVariableOpReadVariableOp.dense_65_matmul_readvariableop_dense_65_kernel*
_output_shapes

:
*
dtype0�
dense_65/MatMulMatMuldense_64/Relu:activations:0&dense_65/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_65/BiasAdd/ReadVariableOpReadVariableOp-dense_65_biasadd_readvariableop_dense_65_bias*
_output_shapes
:*
dtype0�
dense_65/BiasAddBiasAdddense_65/MatMul:product:0'dense_65/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_65/SoftmaxSoftmaxdense_65/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_65/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_63/BiasAdd/ReadVariableOp^dense_63/MatMul/ReadVariableOp ^dense_64/BiasAdd/ReadVariableOp^dense_64/MatMul/ReadVariableOp ^dense_65/BiasAdd/ReadVariableOp^dense_65/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2B
dense_63/BiasAdd/ReadVariableOpdense_63/BiasAdd/ReadVariableOp2@
dense_63/MatMul/ReadVariableOpdense_63/MatMul/ReadVariableOp2B
dense_64/BiasAdd/ReadVariableOpdense_64/BiasAdd/ReadVariableOp2@
dense_64/MatMul/ReadVariableOpdense_64/MatMul/ReadVariableOp2B
dense_65/BiasAdd/ReadVariableOpdense_65/BiasAdd/ReadVariableOp2@
dense_65/MatMul/ReadVariableOpdense_65/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_16256
input_22!
dense_63_kernel:


dense_63_bias:
!
dense_64_kernel:


dense_64_bias:
!
dense_65_kernel:

dense_65_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_22dense_63_kerneldense_63_biasdense_64_kerneldense_64_biasdense_65_kerneldense_65_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_16026o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_22
�
�
(__inference_model_21_layer_call_fn_16278

inputs!
dense_63_kernel:


dense_63_bias:
!
dense_64_kernel:


dense_64_bias:
!
dense_65_kernel:

dense_65_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_63_kerneldense_63_biasdense_64_kerneldense_64_biasdense_65_kerneldense_65_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_model_21_layer_call_and_return_conditional_losses_16171o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�2
�
!__inference__traced_restore_16487
file_prefix2
 assignvariableop_dense_63_kernel:

.
 assignvariableop_1_dense_63_bias:
4
"assignvariableop_2_dense_64_kernel:

.
 assignvariableop_3_dense_64_bias:
4
"assignvariableop_4_dense_65_kernel:
.
 assignvariableop_5_dense_65_bias:1
'assignvariableop_6_training_42_sgd_iter:	 2
(assignvariableop_7_training_42_sgd_decay: :
0assignvariableop_8_training_42_sgd_learning_rate: 5
+assignvariableop_9_training_42_sgd_momentum: &
assignvariableop_10_total_21: &
assignvariableop_11_count_21: 
identity_13��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_63_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_63_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_64_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_64_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_65_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_65_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_42_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_training_42_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_training_42_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_training_42_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_21Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_21Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_221
serving_default_input_22:0���������
<
dense_650
StatefulPartitionedCall:0���������tensorflow/serving/predict:�g
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
J
0
1
2
3
$4
%5"
trackable_list_wrapper
J
0
1
2
3
$4
%5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
+trace_0
,trace_1
-trace_2
.trace_32�
(__inference_model_21_layer_call_fn_16088
(__inference_model_21_layer_call_fn_16267
(__inference_model_21_layer_call_fn_16278
(__inference_model_21_layer_call_fn_16217�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z+trace_0z,trace_1z-trace_2z.trace_3
�
/trace_0
0trace_1
1trace_2
2trace_32�
C__inference_model_21_layer_call_and_return_conditional_losses_16303
C__inference_model_21_layer_call_and_return_conditional_losses_16328
C__inference_model_21_layer_call_and_return_conditional_losses_16230
C__inference_model_21_layer_call_and_return_conditional_losses_16243�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z/trace_0z0trace_1z1trace_2z2trace_3
�B�
 __inference__wrapped_model_16026input_22"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
I
3iter
	4decay
5learning_rate
6momentum"
	optimizer
,
7serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
=trace_02�
(__inference_dense_63_layer_call_fn_16335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z=trace_0
�
>trace_02�
C__inference_dense_63_layer_call_and_return_conditional_losses_16346�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z>trace_0
!:

2dense_63/kernel
:
2dense_63/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Dtrace_02�
(__inference_dense_64_layer_call_fn_16353�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
�
Etrace_02�
C__inference_dense_64_layer_call_and_return_conditional_losses_16364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zEtrace_0
!:

2dense_64/kernel
:
2dense_64/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Ktrace_02�
(__inference_dense_65_layer_call_fn_16371�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
�
Ltrace_02�
C__inference_dense_65_layer_call_and_return_conditional_losses_16382�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zLtrace_0
!:
2dense_65/kernel
:2dense_65/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
M0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_21_layer_call_fn_16088input_22"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_21_layer_call_fn_16267inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_21_layer_call_fn_16278inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_21_layer_call_fn_16217input_22"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_21_layer_call_and_return_conditional_losses_16303inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_21_layer_call_and_return_conditional_losses_16328inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_21_layer_call_and_return_conditional_losses_16230input_22"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_21_layer_call_and_return_conditional_losses_16243input_22"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2training_42/SGD/iter
: (2training_42/SGD/decay
':% (2training_42/SGD/learning_rate
":  (2training_42/SGD/momentum
�B�
#__inference_signature_wrapper_16256input_22"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_63_layer_call_fn_16335inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_63_layer_call_and_return_conditional_losses_16346inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_64_layer_call_fn_16353inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_64_layer_call_and_return_conditional_losses_16364inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_65_layer_call_fn_16371inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_65_layer_call_and_return_conditional_losses_16382inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
^
N	variables
O	keras_api
	Ptotal
	Qcount
R
_fn_kwargs"
_tf_keras_metric
.
P0
Q1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
:  (2total_21
:  (2count_21
 "
trackable_dict_wrapper�
 __inference__wrapped_model_16026p$%1�.
'�$
"�
input_22���������

� "3�0
.
dense_65"�
dense_65����������
C__inference_dense_63_layer_call_and_return_conditional_losses_16346\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� {
(__inference_dense_63_layer_call_fn_16335O/�,
%�"
 �
inputs���������

� "����������
�
C__inference_dense_64_layer_call_and_return_conditional_losses_16364\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� {
(__inference_dense_64_layer_call_fn_16353O/�,
%�"
 �
inputs���������

� "����������
�
C__inference_dense_65_layer_call_and_return_conditional_losses_16382\$%/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
(__inference_dense_65_layer_call_fn_16371O$%/�,
%�"
 �
inputs���������

� "�����������
C__inference_model_21_layer_call_and_return_conditional_losses_16230j$%9�6
/�,
"�
input_22���������

p 

 
� "%�"
�
0���������
� �
C__inference_model_21_layer_call_and_return_conditional_losses_16243j$%9�6
/�,
"�
input_22���������

p

 
� "%�"
�
0���������
� �
C__inference_model_21_layer_call_and_return_conditional_losses_16303h$%7�4
-�*
 �
inputs���������

p 

 
� "%�"
�
0���������
� �
C__inference_model_21_layer_call_and_return_conditional_losses_16328h$%7�4
-�*
 �
inputs���������

p

 
� "%�"
�
0���������
� �
(__inference_model_21_layer_call_fn_16088]$%9�6
/�,
"�
input_22���������

p 

 
� "�����������
(__inference_model_21_layer_call_fn_16217]$%9�6
/�,
"�
input_22���������

p

 
� "�����������
(__inference_model_21_layer_call_fn_16267[$%7�4
-�*
 �
inputs���������

p 

 
� "�����������
(__inference_model_21_layer_call_fn_16278[$%7�4
-�*
 �
inputs���������

p

 
� "�����������
#__inference_signature_wrapper_16256|$%=�:
� 
3�0
.
input_22"�
input_22���������
"3�0
.
dense_65"�
dense_65���������