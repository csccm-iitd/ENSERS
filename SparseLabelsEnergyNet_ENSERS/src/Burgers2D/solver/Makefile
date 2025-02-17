
containerName := EnsersBurgers2d
image := fenics_image

buildimage:
	docker build -t $(image) .

container: 
	docker run -it -v "`pwd`:/home" -w /home --name="$(containerName)" $(image)

startcontainer:
	docker start -ia $(containerName) 

rmcontainer: 
	docker rm -f $(containerName)

stopcontainer:
	docker stop $(containerName)

default: 
	build
