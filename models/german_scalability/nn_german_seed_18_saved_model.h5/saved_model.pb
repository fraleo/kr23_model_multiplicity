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
count_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_18
]
count_18/Read/ReadVariableOpReadVariableOpcount_18*
_output_shapes
: *
dtype0
d
total_18VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_18
]
total_18/Read/ReadVariableOpReadVariableOptotal_18*
_output_shapes
: *
dtype0
�
training_36/SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nametraining_36/SGD/momentum
}
,training_36/SGD/momentum/Read/ReadVariableOpReadVariableOptraining_36/SGD/momentum*
_output_shapes
: *
dtype0
�
training_36/SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_36/SGD/learning_rate
�
1training_36/SGD/learning_rate/Read/ReadVariableOpReadVariableOptraining_36/SGD/learning_rate*
_output_shapes
: *
dtype0
~
training_36/SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_36/SGD/decay
w
)training_36/SGD/decay/Read/ReadVariableOpReadVariableOptraining_36/SGD/decay*
_output_shapes
: *
dtype0
|
training_36/SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_36/SGD/iter
u
(training_36/SGD/iter/Read/ReadVariableOpReadVariableOptraining_36/SGD/iter*
_output_shapes
: *
dtype0	
r
dense_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_56/bias
k
!dense_56/bias/Read/ReadVariableOpReadVariableOpdense_56/bias*
_output_shapes
:*
dtype0
z
dense_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
* 
shared_namedense_56/kernel
s
#dense_56/kernel/Read/ReadVariableOpReadVariableOpdense_56/kernel*
_output_shapes

:
*
dtype0
r
dense_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_55/bias
k
!dense_55/bias/Read/ReadVariableOpReadVariableOpdense_55/bias*
_output_shapes
:
*
dtype0
z
dense_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_55/kernel
s
#dense_55/kernel/Read/ReadVariableOpReadVariableOpdense_55/kernel*
_output_shapes

:

*
dtype0
r
dense_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_54/bias
k
!dense_54/bias/Read/ReadVariableOpReadVariableOpdense_54/bias*
_output_shapes
:
*
dtype0
z
dense_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

* 
shared_namedense_54/kernel
s
#dense_54/kernel/Read/ReadVariableOpReadVariableOpdense_54/kernel*
_output_shapes

:

*
dtype0
{
serving_default_input_19Placeholder*'
_output_shapes
:���������
*
dtype0*
shape:���������

�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19dense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/bias*
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
#__inference_signature_wrapper_14006

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
VARIABLE_VALUEdense_54/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_54/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_55/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_55/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_56/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_56/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEtraining_36/SGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining_36/SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEtraining_36/SGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEtraining_36/SGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEtotal_184keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_184keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_54/kernel/Read/ReadVariableOp!dense_54/bias/Read/ReadVariableOp#dense_55/kernel/Read/ReadVariableOp!dense_55/bias/Read/ReadVariableOp#dense_56/kernel/Read/ReadVariableOp!dense_56/bias/Read/ReadVariableOp(training_36/SGD/iter/Read/ReadVariableOp)training_36/SGD/decay/Read/ReadVariableOp1training_36/SGD/learning_rate/Read/ReadVariableOp,training_36/SGD/momentum/Read/ReadVariableOptotal_18/Read/ReadVariableOpcount_18/Read/ReadVariableOpConst*
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
__inference__traced_save_14191
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_54/kerneldense_54/biasdense_55/kerneldense_55/biasdense_56/kerneldense_56/biastraining_36/SGD/itertraining_36/SGD/decaytraining_36/SGD/learning_ratetraining_36/SGD/momentumtotal_18count_18*
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
!__inference__traced_restore_14237��
�
�
C__inference_model_18_layer_call_and_return_conditional_losses_13993
input_19*
dense_54_dense_54_kernel:

$
dense_54_dense_54_bias:
*
dense_55_dense_55_kernel:

$
dense_55_dense_55_bias:
*
dense_56_dense_56_kernel:
$
dense_56_dense_56_bias:
identity�� dense_54/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall�
 dense_54/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_54_dense_54_kerneldense_54_dense_54_bias*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_13794�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_13809�
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_dense_56_kerneldense_56_dense_56_bias*
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
C__inference_dense_56_layer_call_and_return_conditional_losses_13824x
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_19
�
�
#__inference_signature_wrapper_14006
input_19!
dense_54_kernel:


dense_54_bias:
!
dense_55_kernel:


dense_55_bias:
!
dense_56_kernel:

dense_56_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_19dense_54_kerneldense_54_biasdense_55_kerneldense_55_biasdense_56_kerneldense_56_bias*
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
 __inference__wrapped_model_13776o
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
input_19
�	
�
(__inference_model_18_layer_call_fn_13967
input_19!
dense_54_kernel:


dense_54_bias:
!
dense_55_kernel:


dense_55_bias:
!
dense_56_kernel:

dense_56_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_19dense_54_kerneldense_54_biasdense_55_kerneldense_55_biasdense_56_kerneldense_56_bias*
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
C__inference_model_18_layer_call_and_return_conditional_losses_13921o
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
input_19
�
�
C__inference_model_18_layer_call_and_return_conditional_losses_13829

inputs*
dense_54_dense_54_kernel:

$
dense_54_dense_54_bias:
*
dense_55_dense_55_kernel:

$
dense_55_dense_55_bias:
*
dense_56_dense_56_kernel:
$
dense_56_dense_56_bias:
identity�� dense_54/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall�
 dense_54/StatefulPartitionedCallStatefulPartitionedCallinputsdense_54_dense_54_kerneldense_54_dense_54_bias*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_13794�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_13809�
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_dense_56_kerneldense_56_dense_56_bias*
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
C__inference_dense_56_layer_call_and_return_conditional_losses_13824x
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_54_layer_call_and_return_conditional_losses_13794

inputs7
%matmul_readvariableop_dense_54_kernel:

2
$biasadd_readvariableop_dense_54_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_54_kernel*
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
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_54_bias*
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
�
�
C__inference_model_18_layer_call_and_return_conditional_losses_13921

inputs*
dense_54_dense_54_kernel:

$
dense_54_dense_54_bias:
*
dense_55_dense_55_kernel:

$
dense_55_dense_55_bias:
*
dense_56_dense_56_kernel:
$
dense_56_dense_56_bias:
identity�� dense_54/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall�
 dense_54/StatefulPartitionedCallStatefulPartitionedCallinputsdense_54_dense_54_kerneldense_54_dense_54_bias*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_13794�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_13809�
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_dense_56_kerneldense_56_dense_56_bias*
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
C__inference_dense_56_layer_call_and_return_conditional_losses_13824x
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�

�
C__inference_dense_54_layer_call_and_return_conditional_losses_14096

inputs7
%matmul_readvariableop_dense_54_kernel:

2
$biasadd_readvariableop_dense_54_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_54_kernel*
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
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_54_bias*
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
�
�
(__inference_model_18_layer_call_fn_14028

inputs!
dense_54_kernel:


dense_54_bias:
!
dense_55_kernel:


dense_55_bias:
!
dense_56_kernel:

dense_56_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_54_kerneldense_54_biasdense_55_kerneldense_55_biasdense_56_kerneldense_56_bias*
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
C__inference_model_18_layer_call_and_return_conditional_losses_13921o
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
�
�
(__inference_model_18_layer_call_fn_14017

inputs!
dense_54_kernel:


dense_54_bias:
!
dense_55_kernel:


dense_55_bias:
!
dense_56_kernel:

dense_56_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_54_kerneldense_54_biasdense_55_kerneldense_55_biasdense_56_kerneldense_56_bias*
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
C__inference_model_18_layer_call_and_return_conditional_losses_13829o
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
�

�
C__inference_dense_56_layer_call_and_return_conditional_losses_14132

inputs7
%matmul_readvariableop_dense_56_kernel:
2
$biasadd_readvariableop_dense_56_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_56_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_56_bias*
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
�
�
C__inference_model_18_layer_call_and_return_conditional_losses_13980
input_19*
dense_54_dense_54_kernel:

$
dense_54_dense_54_bias:
*
dense_55_dense_55_kernel:

$
dense_55_dense_55_bias:
*
dense_56_dense_56_kernel:
$
dense_56_dense_56_bias:
identity�� dense_54/StatefulPartitionedCall� dense_55/StatefulPartitionedCall� dense_56/StatefulPartitionedCall�
 dense_54/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_54_dense_54_kerneldense_54_dense_54_bias*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_13794�
 dense_55/StatefulPartitionedCallStatefulPartitionedCall)dense_54/StatefulPartitionedCall:output:0dense_55_dense_55_kerneldense_55_dense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_13809�
 dense_56/StatefulPartitionedCallStatefulPartitionedCall)dense_55/StatefulPartitionedCall:output:0dense_56_dense_56_kerneldense_56_dense_56_bias*
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
C__inference_dense_56_layer_call_and_return_conditional_losses_13824x
IdentityIdentity)dense_56/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_54/StatefulPartitionedCall!^dense_55/StatefulPartitionedCall!^dense_56/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2D
 dense_54/StatefulPartitionedCall dense_54/StatefulPartitionedCall2D
 dense_55/StatefulPartitionedCall dense_55/StatefulPartitionedCall2D
 dense_56/StatefulPartitionedCall dense_56/StatefulPartitionedCall:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_19
�

�
C__inference_dense_55_layer_call_and_return_conditional_losses_13809

inputs7
%matmul_readvariableop_dense_55_kernel:

2
$biasadd_readvariableop_dense_55_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_55_kernel*
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
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_55_bias*
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
�
�
(__inference_dense_55_layer_call_fn_14103

inputs!
dense_55_kernel:


dense_55_bias:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_55_kerneldense_55_bias*
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
C__inference_dense_55_layer_call_and_return_conditional_losses_13809o
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
�"
�
__inference__traced_save_14191
file_prefix.
*savev2_dense_54_kernel_read_readvariableop,
(savev2_dense_54_bias_read_readvariableop.
*savev2_dense_55_kernel_read_readvariableop,
(savev2_dense_55_bias_read_readvariableop.
*savev2_dense_56_kernel_read_readvariableop,
(savev2_dense_56_bias_read_readvariableop3
/savev2_training_36_sgd_iter_read_readvariableop	4
0savev2_training_36_sgd_decay_read_readvariableop<
8savev2_training_36_sgd_learning_rate_read_readvariableop7
3savev2_training_36_sgd_momentum_read_readvariableop'
#savev2_total_18_read_readvariableop'
#savev2_count_18_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_54_kernel_read_readvariableop(savev2_dense_54_bias_read_readvariableop*savev2_dense_55_kernel_read_readvariableop(savev2_dense_55_bias_read_readvariableop*savev2_dense_56_kernel_read_readvariableop(savev2_dense_56_bias_read_readvariableop/savev2_training_36_sgd_iter_read_readvariableop0savev2_training_36_sgd_decay_read_readvariableop8savev2_training_36_sgd_learning_rate_read_readvariableop3savev2_training_36_sgd_momentum_read_readvariableop#savev2_total_18_read_readvariableop#savev2_count_18_read_readvariableopsavev2_const"/device:CPU:0*
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
�2
�
!__inference__traced_restore_14237
file_prefix2
 assignvariableop_dense_54_kernel:

.
 assignvariableop_1_dense_54_bias:
4
"assignvariableop_2_dense_55_kernel:

.
 assignvariableop_3_dense_55_bias:
4
"assignvariableop_4_dense_56_kernel:
.
 assignvariableop_5_dense_56_bias:1
'assignvariableop_6_training_36_sgd_iter:	 2
(assignvariableop_7_training_36_sgd_decay: :
0assignvariableop_8_training_36_sgd_learning_rate: 5
+assignvariableop_9_training_36_sgd_momentum: &
assignvariableop_10_total_18: &
assignvariableop_11_count_18: 
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
AssignVariableOpAssignVariableOp assignvariableop_dense_54_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_54_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_55_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_55_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_56_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_56_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_36_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_training_36_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_training_36_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_training_36_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_18Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_18Identity_11:output:0"/device:CPU:0*
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
_user_specified_namefile_prefix
�
�
C__inference_model_18_layer_call_and_return_conditional_losses_14053

inputs@
.dense_54_matmul_readvariableop_dense_54_kernel:

;
-dense_54_biasadd_readvariableop_dense_54_bias:
@
.dense_55_matmul_readvariableop_dense_55_kernel:

;
-dense_55_biasadd_readvariableop_dense_55_bias:
@
.dense_56_matmul_readvariableop_dense_56_kernel:
;
-dense_56_biasadd_readvariableop_dense_56_bias:
identity��dense_54/BiasAdd/ReadVariableOp�dense_54/MatMul/ReadVariableOp�dense_55/BiasAdd/ReadVariableOp�dense_55/MatMul/ReadVariableOp�dense_56/BiasAdd/ReadVariableOp�dense_56/MatMul/ReadVariableOp�
dense_54/MatMul/ReadVariableOpReadVariableOp.dense_54_matmul_readvariableop_dense_54_kernel*
_output_shapes

:

*
dtype0{
dense_54/MatMulMatMulinputs&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_54/BiasAdd/ReadVariableOpReadVariableOp-dense_54_biasadd_readvariableop_dense_54_bias*
_output_shapes
:
*
dtype0�
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_55/MatMul/ReadVariableOpReadVariableOp.dense_55_matmul_readvariableop_dense_55_kernel*
_output_shapes

:

*
dtype0�
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_55/BiasAdd/ReadVariableOpReadVariableOp-dense_55_biasadd_readvariableop_dense_55_bias*
_output_shapes
:
*
dtype0�
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_55/ReluReludense_55/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_56/MatMul/ReadVariableOpReadVariableOp.dense_56_matmul_readvariableop_dense_56_kernel*
_output_shapes

:
*
dtype0�
dense_56/MatMulMatMuldense_55/Relu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_56/BiasAdd/ReadVariableOpReadVariableOp-dense_56_biasadd_readvariableop_dense_56_bias*
_output_shapes
:*
dtype0�
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_56/SoftmaxSoftmaxdense_56/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_56/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_dense_56_layer_call_fn_14121

inputs!
dense_56_kernel:

dense_56_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_56_kerneldense_56_bias*
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
C__inference_dense_56_layer_call_and_return_conditional_losses_13824o
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
C__inference_dense_56_layer_call_and_return_conditional_losses_13824

inputs7
%matmul_readvariableop_dense_56_kernel:
2
$biasadd_readvariableop_dense_56_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_56_kernel*
_output_shapes

:
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_56_bias*
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
�
�
(__inference_dense_54_layer_call_fn_14085

inputs!
dense_54_kernel:


dense_54_bias:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_54_kerneldense_54_bias*
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
C__inference_dense_54_layer_call_and_return_conditional_losses_13794o
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
�
C__inference_model_18_layer_call_and_return_conditional_losses_14078

inputs@
.dense_54_matmul_readvariableop_dense_54_kernel:

;
-dense_54_biasadd_readvariableop_dense_54_bias:
@
.dense_55_matmul_readvariableop_dense_55_kernel:

;
-dense_55_biasadd_readvariableop_dense_55_bias:
@
.dense_56_matmul_readvariableop_dense_56_kernel:
;
-dense_56_biasadd_readvariableop_dense_56_bias:
identity��dense_54/BiasAdd/ReadVariableOp�dense_54/MatMul/ReadVariableOp�dense_55/BiasAdd/ReadVariableOp�dense_55/MatMul/ReadVariableOp�dense_56/BiasAdd/ReadVariableOp�dense_56/MatMul/ReadVariableOp�
dense_54/MatMul/ReadVariableOpReadVariableOp.dense_54_matmul_readvariableop_dense_54_kernel*
_output_shapes

:

*
dtype0{
dense_54/MatMulMatMulinputs&dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_54/BiasAdd/ReadVariableOpReadVariableOp-dense_54_biasadd_readvariableop_dense_54_bias*
_output_shapes
:
*
dtype0�
dense_54/BiasAddBiasAdddense_54/MatMul:product:0'dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_54/ReluReludense_54/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_55/MatMul/ReadVariableOpReadVariableOp.dense_55_matmul_readvariableop_dense_55_kernel*
_output_shapes

:

*
dtype0�
dense_55/MatMulMatMuldense_54/Relu:activations:0&dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
dense_55/BiasAdd/ReadVariableOpReadVariableOp-dense_55_biasadd_readvariableop_dense_55_bias*
_output_shapes
:
*
dtype0�
dense_55/BiasAddBiasAdddense_55/MatMul:product:0'dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
b
dense_55/ReluReludense_55/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
dense_56/MatMul/ReadVariableOpReadVariableOp.dense_56_matmul_readvariableop_dense_56_kernel*
_output_shapes

:
*
dtype0�
dense_56/MatMulMatMuldense_55/Relu:activations:0&dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_56/BiasAdd/ReadVariableOpReadVariableOp-dense_56_biasadd_readvariableop_dense_56_bias*
_output_shapes
:*
dtype0�
dense_56/BiasAddBiasAdddense_56/MatMul:product:0'dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
dense_56/SoftmaxSoftmaxdense_56/BiasAdd:output:0*
T0*'
_output_shapes
:���������i
IdentityIdentitydense_56/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_54/BiasAdd/ReadVariableOp^dense_54/MatMul/ReadVariableOp ^dense_55/BiasAdd/ReadVariableOp^dense_55/MatMul/ReadVariableOp ^dense_56/BiasAdd/ReadVariableOp^dense_56/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2B
dense_54/BiasAdd/ReadVariableOpdense_54/BiasAdd/ReadVariableOp2@
dense_54/MatMul/ReadVariableOpdense_54/MatMul/ReadVariableOp2B
dense_55/BiasAdd/ReadVariableOpdense_55/BiasAdd/ReadVariableOp2@
dense_55/MatMul/ReadVariableOpdense_55/MatMul/ReadVariableOp2B
dense_56/BiasAdd/ReadVariableOpdense_56/BiasAdd/ReadVariableOp2@
dense_56/MatMul/ReadVariableOpdense_56/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
(__inference_model_18_layer_call_fn_13838
input_19!
dense_54_kernel:


dense_54_bias:
!
dense_55_kernel:


dense_55_bias:
!
dense_56_kernel:

dense_56_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_19dense_54_kerneldense_54_biasdense_55_kerneldense_55_biasdense_56_kerneldense_56_bias*
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
C__inference_model_18_layer_call_and_return_conditional_losses_13829o
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
input_19
� 
�
 __inference__wrapped_model_13776
input_19I
7model_18_dense_54_matmul_readvariableop_dense_54_kernel:

D
6model_18_dense_54_biasadd_readvariableop_dense_54_bias:
I
7model_18_dense_55_matmul_readvariableop_dense_55_kernel:

D
6model_18_dense_55_biasadd_readvariableop_dense_55_bias:
I
7model_18_dense_56_matmul_readvariableop_dense_56_kernel:
D
6model_18_dense_56_biasadd_readvariableop_dense_56_bias:
identity��(model_18/dense_54/BiasAdd/ReadVariableOp�'model_18/dense_54/MatMul/ReadVariableOp�(model_18/dense_55/BiasAdd/ReadVariableOp�'model_18/dense_55/MatMul/ReadVariableOp�(model_18/dense_56/BiasAdd/ReadVariableOp�'model_18/dense_56/MatMul/ReadVariableOp�
'model_18/dense_54/MatMul/ReadVariableOpReadVariableOp7model_18_dense_54_matmul_readvariableop_dense_54_kernel*
_output_shapes

:

*
dtype0�
model_18/dense_54/MatMulMatMulinput_19/model_18/dense_54/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
(model_18/dense_54/BiasAdd/ReadVariableOpReadVariableOp6model_18_dense_54_biasadd_readvariableop_dense_54_bias*
_output_shapes
:
*
dtype0�
model_18/dense_54/BiasAddBiasAdd"model_18/dense_54/MatMul:product:00model_18/dense_54/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
model_18/dense_54/ReluRelu"model_18/dense_54/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
'model_18/dense_55/MatMul/ReadVariableOpReadVariableOp7model_18_dense_55_matmul_readvariableop_dense_55_kernel*
_output_shapes

:

*
dtype0�
model_18/dense_55/MatMulMatMul$model_18/dense_54/Relu:activations:0/model_18/dense_55/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
(model_18/dense_55/BiasAdd/ReadVariableOpReadVariableOp6model_18_dense_55_biasadd_readvariableop_dense_55_bias*
_output_shapes
:
*
dtype0�
model_18/dense_55/BiasAddBiasAdd"model_18/dense_55/MatMul:product:00model_18/dense_55/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
t
model_18/dense_55/ReluRelu"model_18/dense_55/BiasAdd:output:0*
T0*'
_output_shapes
:���������
�
'model_18/dense_56/MatMul/ReadVariableOpReadVariableOp7model_18_dense_56_matmul_readvariableop_dense_56_kernel*
_output_shapes

:
*
dtype0�
model_18/dense_56/MatMulMatMul$model_18/dense_55/Relu:activations:0/model_18/dense_56/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(model_18/dense_56/BiasAdd/ReadVariableOpReadVariableOp6model_18_dense_56_biasadd_readvariableop_dense_56_bias*
_output_shapes
:*
dtype0�
model_18/dense_56/BiasAddBiasAdd"model_18/dense_56/MatMul:product:00model_18/dense_56/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
model_18/dense_56/SoftmaxSoftmax"model_18/dense_56/BiasAdd:output:0*
T0*'
_output_shapes
:���������r
IdentityIdentity#model_18/dense_56/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^model_18/dense_54/BiasAdd/ReadVariableOp(^model_18/dense_54/MatMul/ReadVariableOp)^model_18/dense_55/BiasAdd/ReadVariableOp(^model_18/dense_55/MatMul/ReadVariableOp)^model_18/dense_56/BiasAdd/ReadVariableOp(^model_18/dense_56/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*2
_input_shapes!
:���������
: : : : : : 2T
(model_18/dense_54/BiasAdd/ReadVariableOp(model_18/dense_54/BiasAdd/ReadVariableOp2R
'model_18/dense_54/MatMul/ReadVariableOp'model_18/dense_54/MatMul/ReadVariableOp2T
(model_18/dense_55/BiasAdd/ReadVariableOp(model_18/dense_55/BiasAdd/ReadVariableOp2R
'model_18/dense_55/MatMul/ReadVariableOp'model_18/dense_55/MatMul/ReadVariableOp2T
(model_18/dense_56/BiasAdd/ReadVariableOp(model_18/dense_56/BiasAdd/ReadVariableOp2R
'model_18/dense_56/MatMul/ReadVariableOp'model_18/dense_56/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������

"
_user_specified_name
input_19
�

�
C__inference_dense_55_layer_call_and_return_conditional_losses_14114

inputs7
%matmul_readvariableop_dense_55_kernel:

2
$biasadd_readvariableop_dense_55_bias:

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp{
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_55_kernel*
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
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_55_bias*
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

 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
input_191
serving_default_input_19:0���������
<
dense_560
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
(__inference_model_18_layer_call_fn_13838
(__inference_model_18_layer_call_fn_14017
(__inference_model_18_layer_call_fn_14028
(__inference_model_18_layer_call_fn_13967�
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
C__inference_model_18_layer_call_and_return_conditional_losses_14053
C__inference_model_18_layer_call_and_return_conditional_losses_14078
C__inference_model_18_layer_call_and_return_conditional_losses_13980
C__inference_model_18_layer_call_and_return_conditional_losses_13993�
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
 __inference__wrapped_model_13776input_19"�
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
(__inference_dense_54_layer_call_fn_14085�
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
C__inference_dense_54_layer_call_and_return_conditional_losses_14096�
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
2dense_54/kernel
:
2dense_54/bias
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
(__inference_dense_55_layer_call_fn_14103�
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
C__inference_dense_55_layer_call_and_return_conditional_losses_14114�
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
2dense_55/kernel
:
2dense_55/bias
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
(__inference_dense_56_layer_call_fn_14121�
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
C__inference_dense_56_layer_call_and_return_conditional_losses_14132�
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
2dense_56/kernel
:2dense_56/bias
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
(__inference_model_18_layer_call_fn_13838input_19"�
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
(__inference_model_18_layer_call_fn_14017inputs"�
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
(__inference_model_18_layer_call_fn_14028inputs"�
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
(__inference_model_18_layer_call_fn_13967input_19"�
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
C__inference_model_18_layer_call_and_return_conditional_losses_14053inputs"�
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
C__inference_model_18_layer_call_and_return_conditional_losses_14078inputs"�
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
C__inference_model_18_layer_call_and_return_conditional_losses_13980input_19"�
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
C__inference_model_18_layer_call_and_return_conditional_losses_13993input_19"�
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
:	 (2training_36/SGD/iter
: (2training_36/SGD/decay
':% (2training_36/SGD/learning_rate
":  (2training_36/SGD/momentum
�B�
#__inference_signature_wrapper_14006input_19"�
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
(__inference_dense_54_layer_call_fn_14085inputs"�
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
C__inference_dense_54_layer_call_and_return_conditional_losses_14096inputs"�
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
(__inference_dense_55_layer_call_fn_14103inputs"�
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
C__inference_dense_55_layer_call_and_return_conditional_losses_14114inputs"�
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
(__inference_dense_56_layer_call_fn_14121inputs"�
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
C__inference_dense_56_layer_call_and_return_conditional_losses_14132inputs"�
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
:  (2total_18
:  (2count_18
 "
trackable_dict_wrapper�
 __inference__wrapped_model_13776p$%1�.
'�$
"�
input_19���������

� "3�0
.
dense_56"�
dense_56����������
C__inference_dense_54_layer_call_and_return_conditional_losses_14096\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� {
(__inference_dense_54_layer_call_fn_14085O/�,
%�"
 �
inputs���������

� "����������
�
C__inference_dense_55_layer_call_and_return_conditional_losses_14114\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� {
(__inference_dense_55_layer_call_fn_14103O/�,
%�"
 �
inputs���������

� "����������
�
C__inference_dense_56_layer_call_and_return_conditional_losses_14132\$%/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� {
(__inference_dense_56_layer_call_fn_14121O$%/�,
%�"
 �
inputs���������

� "�����������
C__inference_model_18_layer_call_and_return_conditional_losses_13980j$%9�6
/�,
"�
input_19���������

p 

 
� "%�"
�
0���������
� �
C__inference_model_18_layer_call_and_return_conditional_losses_13993j$%9�6
/�,
"�
input_19���������

p

 
� "%�"
�
0���������
� �
C__inference_model_18_layer_call_and_return_conditional_losses_14053h$%7�4
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
C__inference_model_18_layer_call_and_return_conditional_losses_14078h$%7�4
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
(__inference_model_18_layer_call_fn_13838]$%9�6
/�,
"�
input_19���������

p 

 
� "�����������
(__inference_model_18_layer_call_fn_13967]$%9�6
/�,
"�
input_19���������

p

 
� "�����������
(__inference_model_18_layer_call_fn_14017[$%7�4
-�*
 �
inputs���������

p 

 
� "�����������
(__inference_model_18_layer_call_fn_14028[$%7�4
-�*
 �
inputs���������

p

 
� "�����������
#__inference_signature_wrapper_14006|$%=�:
� 
3�0
.
input_19"�
input_19���������
"3�0
.
dense_56"�
dense_56���������