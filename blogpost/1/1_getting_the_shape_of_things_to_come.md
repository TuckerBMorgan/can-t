# So you want to write your own Pytorch
### Post \#1  Getting the Shape of Things to Come 

So for some reason you have decided that Pytorch/Tensor/JAX are not for you, and you want to write your own, well here is a guide.

I started this project myself after notcing everythig time I went to go learn ML I just got sidetracked writing my own version of Pytorch. So I just dedcided to commit to it honestly. 

I had a few goals when I set out to start writing this.

1. Make choices that maximize learning over about anything else. 
    This is a choice that has helped me not get bogged down on personal projects before, so I made it here as well, so if you read someone and say to yourself "there is much better way of doing this" often the answear was "Yeah, but I would likely leraning less, or get bogged some writing it for six months"

2. Over comment, and over test. I am not always the greatest about either of these, and so I saw this as a good project to go to an extreme with them. To me I think I might be going too far on comments, and still undertesting, but what is the right amount of both is a question that can grind any engineering team to a halt. 

3. Minimal AI Code. I wrote the first version of this lib, Poro[LINK], partially in Cursor, and I like many people could not belive how much faster I went. But then, I found bugs, I had to add functionality to old code, and I found that often I did not understand the code base. So with Cant I started over and decided that I would use a basic old text editor(VSCODE) and not use any AI coding assit. I still use ChatGPT, and I find it useful for plenty of things, I just want to make sure I understand what is going on in this code base day to day.

Nothing to exciting, but they have helped shape this project in profound ways and had made it easier to come back day after day, each feeling like I am actually growing, and enjoying my time with this code base.


With these posts I want to focus on a facet of this code base, and hopefuly help someone else down the road.


So what is Cant, generally. It is a ML library, a Tensor library, a pytorch like? There has yet to be a single term for a library like this, both ML library and Tensor library feel like they are missing someone crucial. I am going to use the term Finkel, because why not, moving on. 

Almost all of the Finkels tend to have a few common parts and that is what this post is going to cover. Those common operations are

1. Tensors, which you can think of as matrices with extra dimensions
2. Custom Tensor to Tensor opeartions. Like Add, Multiply, and Matrix Multiplication
3. Shape. A tracker of some kind that defines how we view the tensor

And beyond that we start getting into "choices" and those choices I will cover in a future post.

For the most part I want to focus on Shape, which might at first seem like a basic and unimportant concept, but for my first way through with Poro, taking that approach led to plenty of problems that I made sure to correect with Cant.

Basicaly when it comes to a matrix it is importnat to know it dimensions.

If it just one element then you would say it has a shape of [1]

two elements [2]

but we can get extra spicy, by adding another dimension, so a matrix with Shape [2, 2] would look like something like
[[1, 2], [3, 4]]

// ADD A SECTION ABOUT GETTING SUB MATRICES

You can keeping adding dimensions forever, but with Cant we only go to 4. Why 4? well because that is the most we would ever need, since I am only planning on getting to gpt2, which at most has a 4d matrix. 


