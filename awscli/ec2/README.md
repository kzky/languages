# AWS CLI Reference
- https://docs.aws.amazon.com/cli/latest/reference/ec2/index.html


# CLI examples

**Tips**

There is `--dry-run` option to confirm your intended command is correct or not. So, put `--dry-run` option just at the first time when you run a command.


## Working from Template

### Launch

```bash
aws ec2 run-instances --profile <profile> \
  --launch-template 'LaunchTemplateId=<template-id>,Version=<N>' \
  --count 1 \
  --instance-type p3.2xlarge \
  --key-name <profile> \
  --tag-specification 'ResourceType=instance,Tags=[{Key=Name,Value=<YourName>-<SMT>},{Key=Nightly,Value=True}]' \
  --block-device-mappings '[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 75, "DeleteOnTermination": true}}]'
```


### Launch with additional EBS

```bash
aws ec2 run-instances --profile <profile> \
  --launch-template 'LaunchTemplateId=<template-id>,Version=<N>' \
  --count 1 \
  --instance-type p3.2xlarge \
  --key-name <profile> \
  --tag-specification 'ResourceType=instance,Tags=[{Key=Name,Value=<instance-name>},{Key=Nightly,Value=True}]' \
  --block-device-mappings '[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 50, "DeleteOnTermination": true}},{"DeviceName": "/dev/sdb", "Ebs": {"VolumeSize": 250, "DeleteOnTermination": true, "VolumeType": "gp2"}}]'
```

### SSH login

```bash
ssh -i ~/.ssh/<profile>.pem ubuntu@<ip-addr>
```

### Scp

```bash
scp -6 -i ~/.ssh/<profile>.pem /usr/local/bin/run-docker-image ubuntu@<ip-addr>:
```

### Describe instance

```bash
aws ec2 describe-instances --filters "Name=tag-key,Values=Billing" "Name=tag-value,Values=<value>"
```

### Stop instance

```bash
aws ec2 stop-instances --profile <profile> --instance-ids <instance-id>
```

### Modify instance

```bash
aws ec2 modify-instance-attribute --profile <profile> --instance-id <instance-id> --instance-type <instance-type>
```

```bash
aws ec2 create-tags --profile <profile> --resources <instance-id or ami-id> --tags '[{"Key": "Nightly","Value": "True"}]'
```

### Start instance

```bash
aws ec2 start-instances --profile <profile> --instance-ids <instance-ids>
```


### Terminate instance

```bash
aws ec2 terminate-instances --profile <profile> --instance-ids <instance-id>
```

### Create AMI

```bash
aws ec2 create-image --profile <profile> --instance-id <instance-id> --name <ami-name>
```

