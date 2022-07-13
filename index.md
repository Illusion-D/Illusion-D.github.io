[TOC]



# 数据结构

## 基础数据结构

### 栈

先进去的后出来

### 队列

先进先出

### 链表



------

## 进阶数据结构

### 树状数组



------



### 线段树

#### 基本概念

**线段树是基于分治思想的二叉树。**

为了引入线段树，我们来看一个例子：

**例**	给你一个序列$a$, 需要支持一下几种操作

​		1.查询区间$[l, r]$上的值的和；

​		2.修改某一个位置上的值；

​		3.区间$[l, r]$值$+k$ ;

其实这是可以用前缀和来做的，但未免慢了些；前两个操作树状数组可以很简洁地完成，但第三个就有些麻烦了，而且效率也是不及线段树的。

类似于树状数组，线段树也是一个节点管理多个原数组的数的一些信息。

线段树的**特点**：

1. 每个节点代表一个区间；
2. 线段树具有唯一的根节点，代表区间是整个统计范围（可持久化除外，有多个）；
3. 线段树的每个叶子节点都代表一个长度为1的原区间$[x, x]$;
4. 对于每个节点$[l, r]$ , **其左儿子是$[l, mid]$, 右儿子是$[mid + 1, r]$, 其中 $mid = (l + r)/2$;**

如图：

<img src="E:\拾荒记\图片\segmentree.png" alt="线段树" style="zoom:50%;" />

#### 普通线段树的实现

##### 线段树的存储

线段树是一种二叉树，所以我们可以采用父子二倍的方法来实现器存储。对于一个节点 $p$ ,其左儿子是 $p*2$ ,右儿子是 $p*2+1$ 。

对于每个节点，都会存储一些信息，所以我们可以用一个结构体来存：

```cpp
struct node{
    int l, r;//这个节点所维护的区间
    int dat;//这个节点所维护的区间的信息
}tree[4*maxn];
```

当然，也可以不用结构体，只要在调用相关函数时上传$[l, r]$ 信息即可。

我们可以看到，线段树数组开的空间是$4*maxn$ ，这是为什么呢。首先我们可以从上图看到，对于整个区间长度是二的n次幂的区间，需要原数组二倍的空间（$2n-1$) ,而如果不是二的n次幂呢？那就会多一层，而多的那一层的所占空间是$2n$ ,而且我们采用的父子二倍的方法，所以就算最后一层没有完全占用，依旧要开上它的空间。

##### 建树

在输入原数组后，就可以先建树了，建完树后进行一些区间的维护也方便。

建树是从根节点开始的，层层向下递归，直到遇到叶子节点后保存叶子节点信息，再向上回溯维护信息。

```cpp
void build(int p, int l, int r){
    tree[p].l = l, tree[p].r = r;
    if(l == r){				//遇到叶子节点
        tree[p].dat = a[l];
        return;
    }
    int mid = l + r >> 1;
    build(p * 2, l, mid);		//建左子树
    build(p * 2 + 1, mid + 1, r);	//建右子树
    tree[p].dat = tree[p * 2].dat + tree[p * 2 + 1].dat; //维护信息
}


//调用：
build(1, 1, n);
```

##### 单点修改与区间查询



##### 区间修改

需要用到一个懒惰标记，这样就不用每次修改都把值传下去，而是在后续操作中遇到标记再下传。

模板：[P3372 【模板】线段树 1](https://www.luogu.com.cn/problem/P3372)

```cpp
#include <bits/stdc++.h>
#define maxn 100000
#define int long long
using namespace std;
int m,n;
#define ls k << 1
#define rs k << 1 | 1
#define mid (l + r >> 1)
int tree[maxn << 2], tag[maxn << 2];
void pushup(int k) {
	tree[k] = tree[ls] + tree[rs];
}
void Pushup(int k) {
	tree[k] = tree[ls] + tree[rs];
}
void Build(int k, int l, int r) {
	if (l == r) {
		cin >> tree[k];
		return;
	}
	Build(ls, l, mid);
	Build(rs, mid + 1, r);
	pushup(k);
}
void Add(int k, int l, int r, int v) {
	tag[k] += v;
	tree[k] += 1ll * (r - l + 1) * v;
}
void pushdown(int k, int l, int r) {
	if (!tag[k]) return;
	Add(ls, l, mid, tag[k]);
	Add(rs, mid + 1, r, tag[k]);
	tag[k] = 0;
}
void Modify(int k, int l, int r, int x, int y, int v) {
	if (l >= x && r <= y) return Add(k, l, r, v);
	pushdown(k, l, r);
	if (x <= mid) Modify(ls, l, mid, x, y, v);
	if (mid < y) Modify(rs, mid + 1, r, x, y, v);
	pushup(k);
}
int Query(int k, int l, int r, int x, int y) {
	if (l >= x && r <= y) return tree[k];
	pushdown(k, l, r);
	int ret = 0;
	if (x <= mid) ret += Query(ls, l, mid, x, y);
	if (mid < y) ret += Query(rs, mid + 1, r, x, y);
	return ret;
}
signed main() {
	cin >> n >> m;
	Build(1, 1, n);
	for (int i = 1, opt, x, y, z; i <= m; i++) {
		cin >> opt >> x >> y;
		if (opt == 1) {
			cin >> z;
			Modify(1, 1, n, x, y, z);
		}
		if(opt == 2){
			cout<<Query(1, 1, n, x, y)<<endl;
		}
	}
	return 0;
}

```

区间乘法：需要维护两个懒惰标记，一个加一个乘，在下传时要注意顺序。

模板：[P3373 【模板】线段树 2](https://www.luogu.com.cn/problem/P3373)

```cpp
#include<bits/stdc++.h>
#define maxn 100005
#define int long long
using namespace std;
int n, m, mod;
int a[maxn];
struct node{
	int l, r, tag1, tag2, sum;
}t[4*maxn];
void build(int p, int l, int r){
	t[p].l = l, t[p].r = r, t[p].tag2 = 1;
	if(l == r){
		t[p].sum = a[l]%mod;
		return;
	}
	int mid = l + r >> 1;
	build(p*2, l, mid);
	build(p*2+1, mid + 1, r);
	t[p].sum = (t[p*2].sum + t[p*2+1].sum) %mod;
}
void spread(int p){
	t[p*2].sum = (t[p].tag2 * t[p*2].sum + (t[p*2].r - t[p*2].l + 1) * t[p].tag1)%mod ;
    t[p*2+1].sum = (t[p].tag2 * t[p*2+1].sum + (t[p].tag1 * (t[p*2+1].r - t[p*2+1].l + 1)) )%mod ;
    t[p*2].tag2 = (t[p*2].tag2 * t[p].tag2)%mod ;
    t[p*2+1].tag2 = (t[p*2+1].tag2 * t[p].tag2)%mod ;
	t[p*2].tag1 = (t[p*2].tag1 * t[p].tag2 + t[p].tag1)%mod ;
    t[p*2+1].tag1 = (t[p*2+1].tag1 * t[p].tag2 + t[p].tag1) %mod;
    t[p].tag2 = 1;
	t[p].tag1 = 0;
}
void add(int p, int x, int y, int k){
	if(x <= t[p].l && y >= t[p].r){
		t[p].sum = (t[p].sum + k * (t[p].r - t[p].l + 1)) % mod;
		t[p].tag1 = (t[p].tag1 + k)%mod ;
		return;
	}
	spread(p);
	t[p].sum = (t[p*2].sum + t[p*2+1].sum) ;
	int mid = t[p].l + t[p].r >> 1;
	if(x <= mid) add(p*2, x, y, k);
	if(y > mid) add(p*2+1, x, y, k);
	t[p].sum = (t[p*2].sum + t[p*2+1].sum) ;
}
void mul(int p,int x,int y,int k){
	if(t[p].l>=x && t[p].r<=y){
		t[p].tag1 = (t[p].tag1 * k) ;
		t[p].tag2 = (t[p].tag2 * k) ;
		t[p].sum = (t[p].sum * k) ;
		return ;
	}
	spread(p);
    t[p].sum = t[p*2].sum + t[p*2+1].sum;
	int mid = (t[p].l + t[p].r) >> 1;
	if(x <= mid) mul(p*2, x, y, k);
	if(mid < y) mul(p*2+1, x, y, k);
	t[p].sum = (t[p*2].sum + t[p*2+1].sum)%mod;
}
int ask(int p, int x, int y){
	if(t[p].l >= x && t[p].r <= y){
		return t[p].sum;
	}
	spread(p);
	int as=0;
	int mid = (t[p].l+t[p].r)>>1;
	if(x <= mid) as = (as + ask(p*2, x, y))%mod ;
	if(mid < y) as = (as + ask(p*2+1, x, y))%mod ;
	return as;
}
signed main(){
	cin >> n >> m >> mod;
	for(int i = 1; i <= n; i++) cin >> a[i];
	build(1, 1, n);
	for(int i = 1, opt, x, y, k; i <= m; i++){
		cin >> opt;
		if(opt == 1){
			cin >> x >> y >> k;
			mul(1, x, y, k);
		}
		if(opt == 2){
			cin >> x >> y >> k;
			add(1, x, y, k);
		}
		if(opt == 3){
			cin >> x >>y;
			cout << ask(1, x, y)%mod << endl;
		}
	}
	return 0;
} 
```



#### 线段树应用

##### 扫描线

##### 权值线段树

所谓权值线段树就是在值域上建一棵线段树，当插入一个数时，其位置加一。

**例题：**[ [USACO08FEB]Hotel G](https://www.luogu.com.cn/problem/P2894) 





**优化：动态开点**

我们可以看到，在值域上开一棵线段树是非常危险的，这很可能会爆空间



#### 可持久化线段树

可持久化线段树即为可保存历史版本的线段树。

在一个历史版本上修改时，只用新建被修改了的节点，没有修改的节点直接指向原来的历史版本就行。

根节点仍然是调用这棵树的入口，会有很多根节点，每一个根节点都是其所在的历史版本的入口。

##### 可持久化数组

模板：[P3919 【模板】可持久化线段树 1（可持久化数组）](https://www.luogu.com.cn/problem/P3919)

```cpp
#include <bits/stdc++.h>
#define maxn 1000012
using namespace std;
struct node{
	int lc, rc, dat;
}t[30*maxn];
int root[maxn], tot;
int n, m, a[maxn], cnt;
inline int read(){
    int f=1,x=0;char ch;
    do{ch=getchar();if(ch=='-')f=-1;}while(ch<'0'||ch>'9');
    do{x=x*10+ch-'0';ch=getchar();}while(ch>='0'&&ch<='9');
    return f*x;
}
int build(int l, int r){
	int p = ++tot;
	if(l == r){
		t[p].dat = a[l];
		return p;
	}
	int mid = (l + r) >> 1;
	t[p].lc = build(l, mid);
	t[p].rc = build(mid + 1, r);
	t[p].dat = max(t[t[p].lc].dat , t[t[p].rc].dat);
	return p;
}
int change(int now, int l, int r, int x, int val){
	int p = ++tot;
	t[p] = t[now];
	if(l == r){
		t[p].dat = val;
		return p;
	}
	int mid = (l + r) >> 1;
	if(x <= mid) t[p].lc = change(t[now].lc, l, mid, x, val);
	else t[p].rc = change(t[now].rc, mid+1, r, x, val);
	t[p].dat = max(t[t[p].lc].dat , t[t[p].rc].dat);
	return p;
}
int query(int p, int l, int r, int x){
	if (l == r) return t[p].dat;
	int mid = (l + r) >> 1;
	if (x <= mid) return query(t[p].lc, l, mid, x);
	else return query(t[p].rc, mid + 1, r, x);
}
signed main(){
	n = read(), m = read();
	for(int i = 1; i <= n; i++) a[i] = read();
	root[0] = build(1, n);
	for(int i = 1, now, opt, x, val; i <= m; i++){
		now = read(), opt = read();
		if(opt == 1){
			x = read(), val = read();
			root[i] = change(root[now], 1, n, x, val);
		}
		if(opt == 2){
			x = read();
			cout << query(root[now], 1, n, x) <<endl;
			root[i] = root[now];
		}
	}
}
```



### 并查集

##### 基本并查集

```cpp
//查询
int find(int k){
    if(f[k] == k) return k;
    return f[k] = find(f[k]);
}
//合并
int merge(int x, y){
    f[find(x)] = find(y);
}
```



##### 并查集进阶应用

**边带权** （模板：[P1196 [NOI2002] 银河英雄传说](https://www.luogu.com.cn/problem/P1196) ）

在本道题中要查询两个战舰之间所隔战舰，整个结构是许多条链，而每条链其实也是一颗特殊的树，可以用边带权的并查集来做。

让同一列每两个战舰之间距离为 1， 则在路径压缩查找时可以顺便统计出每个战舰到祖先的距离，在求



### 平衡树

##### Treap

Treap 是一种 **弱平衡** 的 **二叉搜索树**。它的数据结构由二叉树和二叉堆组合形成，名字也因此为 tree 和 heap 的组合。

Treap 的每个结点上除了按照二叉搜索树排序的 $key$ 值外要额外储存一个叫 $property$ 的值。它由每个结点建立时随机生成，并按照 **最大堆** 性质排序。因此 treap 除了要满足二叉搜索树的性质之外，还需满足父节点的 $property$ 大于等于两个子节点的值。所以它是 **期望平衡** 的。搜索，插入和删除操作的期望时间复杂度为 $O(logn)$。

模板：[P3369 【模板】普通平衡树](https://www.luogu.com.cn/problem/P3369)

```cpp
#include<bits/stdc++.h>
#define int long long
using namespace std;
const int SIZE = 1000000;
struct treap{
	int l,r;
	int val,dat;
	int cnt,size;
}a[SIZE];
int tot,root,n,INF=1e9;
int New(int val){
	a[++tot].val=val;
	a[tot].dat=rand();
	a[tot].cnt=a[tot].size=1;
	return tot;
}
void update(int p){
	a[p].size=a[a[p].l].size+a[a[p].r].size+a[p].cnt;
}
void build(){
	New(-INF),New(INF);
	root=1,a[1].r=2;
	update(root);
}
int getrbyv(int p,int val){
	if(p==0)return 0;
	if(val==a[p].val)return a[a[p].l].size+1;
	if(val<a[p].val)return getrbyv(a[p].l,val);
	return getrbyv(a[p].r,val)+a[a[p].l].size+a[p].cnt;
}
int getvbyr(int p,int rank){
	if(p==0)return 	INF;
	if(a[a[p].l].size>=rank)return getvbyr(a[p].l,rank);
	if(a[a[p].l].size+a[p].cnt>=rank)return a[p].val;
	return getvbyr(a[p].r,rank-a[a[p].l].size-a[p].cnt);
}
void zig(int &p){
	int q=a[p].l;
	a[p].l=a[q].r,a[q].r=p,p=q;
	update(a[p].r),update(p);
}
void zag(int &p){
	int q=a[p].r;
	a[p].r=a[q].l,a[q].l=p,p=q;
	update(a[p].l),update(p);
}
void insert(int &p,int val){
	if(p==0){
		p=New(val);
		return;
	}
	if(val==a[p].val){
		a[p].cnt++,update(p);
		return ;
	}
	if(val<a[p].val){
		insert(a[p].l,val);
		if(a[p].dat<a[a[p].l].dat)zig(p);
	}
	else{
		insert(a[p].r,val);
		if(a[p].dat<a[a[p].r].dat)zag(p);
	}
	update(p);
}
int getpre(int val){
	int ans=1;
	int p=root;
	while(p){
		if(val==a[p].val){
			if(a[p].l>0){
				p=a[p].l;
				while(a[p].r>0)p=a[p].r;
				ans=p;
			}
			break;
		}
		if(a[p].val<val&&a[p].val>a[ans].val)ans=p;
		p=val<a[p].val?a[p].l:a[p].r;
	}
	return a[ans].val;
}
int getnext(int val){
	int ans=2;
	int p=root;
	while(p){
		if(val==a[p].val){
			if(a[p].r>0){
				p=a[p].r;
				while(a[p].l>0)p=a[p].l;
				ans=p;
			}
			break;
		}
		if(a[p].val>val&&a[p].val<a[ans].val) ans=p;
		p=val<a[p].val?a[p].l:a[p].r;
	}
	return a[ans].val;
}
void remove(int &p,int val){
	if(p==0)return ;
	if(val==a[p].val){
		if(a[p].cnt>1){
			a[p].cnt--,update(p);
			return ;
		}
		if(a[p].l||a[p].r){
			if(a[p].r==0||a[a[p].l].dat>a[a[p].r].dat)
				zig(p),remove(a[p].r,val);
			else
				zag(p),remove(a[p].l,val);
			update(p);
		}
		else p=0;
		return ;
	}
	if(val<a[p].val)remove(a[p].l,val);
	else remove(a[p].r,val);
	update(p);
}
signed main(){
	build();
	cin>>n;
	while(n--){
		int opt,x;
		cin>>opt>>x;
		switch(opt){
			case 1: insert(root,x); break;
			case 2: remove(root,x); break;
			case 3:cout<<getrbyv(root,x)-1<<endl;break;
			case 4:cout<<getvbyr(root,x+1)<<endl;break;
			case 5:cout<<getpre(x)<<endl;break;
			case 6:cout<<getnext(x)<<endl;break;
		}
	}
	return 0;		
} 
```



##### 笛卡尔树

笛卡尔树是一种二叉树，每一个结点由一个键值二元组 构成。要求 满足二叉搜索树的性质，而 满足堆的性质。一个有趣的事实是，如果笛卡尔树的 键值确定，且 互不相同， 互不相同，那么这个笛卡尔树的结构是唯一的。上图：

<img src="E:\拾荒记\图片\笛卡尔树1.png" style="zoom:50%;" />



上面这棵笛卡尔树相当于把数组元素值当作键值 ，而把数组下标当作键值 。显然可以发现，这棵树的键值 满足二叉搜索树的性质，而键值 满足小根堆的性质。

其实图中的笛卡尔树是一种特殊的情况，因为二元组的键值 恰好对应数组下标，这种特殊的笛卡尔树有一个性质，就是一棵子树内的下标是连续的一个区间（这样才能满足二叉搜索树的性质）。更一般的情况则是任意二元组构建的笛卡尔树。

如下图建树：

![](E:\拾荒记\图片\cartesian-tree2.webp.jpg)

模板：[P5854 【模板】笛卡尔树](https://www.luogu.com.cn/problem/P5854)

```cpp
#include <bits/stdc++.h>
#define maxn 10000005
using namespace std;
int a[maxn], n; 
struct tree{
	int lc, rc, v;
}t[maxn];
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
} 
int main(){
	n = read();
	int cnt = 0;
	int pos = 0;
	for(int i = 1; i <= n; i++){
		a[i] = read();
		pos = cnt;
		while(pos && a[t[pos].v] > a[i]) pos --;
		if(pos) t[t[pos].v].rc = i;
		if(pos < cnt) t[i].lc = t[pos + 1].v;
		t[cnt = ++pos].v = i;
	}
	long long L=0, R=0;
	for(int i = 1; i <= n; i++){
		L ^=1ll* i * (t[i].lc + 1);
		R ^=1ll* i * (t[i].rc + 1);
	}
	cout << L << " " << R;
} 
```



# 图论

>  **图论 (Graph theory)** 是数学的一个分支，图是图论的主要研究对象。**图 (Graph)** 是由若干给定的顶点及连接两顶点的边所构成的图形，这种图形通常用来描述某些事物之间的某种特定关系。顶点用于代表事物，连接两顶点的边则用于表示两个事物间具有这种关系。 ——摘自 [oi.wiki](oi.wiki)

## 图的建立与存储

##### 1.直接存边

直接存边多用于求最小生成树，它的优点就是可以排序，缺点则是时各边之间没有了连通性，也就是不能遍历。建图时，只需直接将这个边的起点、终点和边权即可：

```cpp
struct edge{
    int u, v, w;
}e[maxn];
输入时：
for(int i = 1; i <= n; i++){
    int u, v, w;
    cin >> u >> v >> w;
    e[i].u = u, e[i].v = v, e[i].w = w;
}
```

##### 2.邻接矩阵

也是存边的一种方法，建立一个二维数组，`a[i][j]`即表示从 `i`到 `j`有一条边，边权为 `a[i][j]` 。

```c++
int a[maxn][maxn];
for(int i = 1; i <= n; i++){
    int u, v, w;
    cin >> u >> v >> w;
    a[u][v] = w;
    //若为无向图，则要加上 a[v][u] = w;
}
```

但是我们可以看到，邻接矩阵是一个二维的，一旦数据过大就不适合使用了。

##### 3.邻接表

对于上述的邻接矩阵，有一种方法可以使其空间复杂度降低的方法，那就是邻接表，即用可变数组的方式存储图，`e[i][j] = a` 的意思即为i有一条连向a的边。

```c++
vector<int> e[maxn];

for(int i = 1, u, v; i <= m; i++){
    cin >> u >> v;
    e[u].push_back(v);
    //无向图：e[v].push_back(u);
}
```

这样，一维固定，一维不固定，便大大节省了空间。

在遍历时，只需：

```c++
void dfs(int x){
    vis[x] = 1;
    for(int i = 0; i < e[x].size(); i++){
        if(vis[e[x][i]]) continue;
        dfs(e[x][i]);
    }
}
```

然而，我们可以发现一个问题，这种方法不能存储边权，而且一旦图比较稠密，空间复杂度依旧很高。

##### 4.链式前向星

链式前向星是用数组的形式实现了一个静态链表。

```c++
struct edge{
    int v, w, nxt;
}e[maxn];
int head[maxn], cnt;

void add(int u, int v, int w){
    e[++cnt].v = v, e[cnt].w = w;
    e[cnt].nxt = head[u], head[u] = cnt;
}
```

遍历：

```cpp
void dfs(int x){
	vis[x] = 1;
	for(int i = head[x]; i; i = e[i].nxt){
		if(vis[e[i].v]) continue;
        dfs(e[i])
	}
}
```



## 最小生成树

### Kruskal算法

本质是将各个边按照边权排序，从小到大依次加入，如果两个点已经联通，则不加这条边。用排序+并查集即可。

<img src="E:\拾荒记\图片\mst-2.png" style="zoom: 50%;" />

例题：[P3366 【模板】最小生成树](https://www.luogu.com.cn/problem/P3366)

给出一个无向图，求出最小生成树，如果该图不连通，则输出 `orz`。

```cpp
#include<bits/stdc++.h>
#define maxn 200005
using namespace std;
int n, m, as;
int f[maxn];
struct node{
	int u, v, w;
}a[maxn];
bool cmp(node x, node y){
	return x.w < y.w;
}
int find(int k){
    if(f[k]==k)return k;
    return f[k]=find(f[k]);
}
int main(){
	int tot = 0;
	cin >> n >> m;
	for(int i = 1; i <= n; i++) f[i] = i;
	for(int i = 1, u, v, w; i <= m; i++){
		cin >> u >> v >> w;
		a[i].u = u, a[i].v = v, a[i].w = w;
	}
	sort(a+1, a + m + 1, cmp);
	for(int i = 1; i <= m; i++){
		if(find(a[i].u) == find(a[i].v)) continue;
		as += a[i].w;
		f[find(a[i].v)] = find(a[i].u);
		if(++tot == n-1) break;
	}
	if(tot != n-1){
		cout << "orz";
		return 0;
	}
	cout << as;
}
```

### Prim算法

不同于Kruskal的加边，prim算法是从一个节点开始不断地加点。

<img src="E:\拾荒记\图片\mst-3.png" style="zoom:50%;" />

模板还是P3366，给出prim的代码：

```cpp
#include<bits/stdc++.h>
using namespace std;
int n,m,cnt,elast[2000005],vis[10000];
struct edge{
	int v,w,next;
}e[2000005];
void add(int x,int y,int z){
	e[cnt].v=y;
	e[cnt].w=z;
	e[cnt].next=elast[x];
	elast[x]=cnt++;
}
int prim(int x){
	priority_queue<pair<int,int> > q;
	for(int i=elast[x];~i;i=e[i].next){
		int j=e[i].v;
		q.push({-e[i].w,j});
	}
	vis[x]=1;
	int ans=0,cntt=0;
	while(q.size()){
		pair<int,int> temp=q.top();
		q.pop();
		int node=temp.second,value=-temp.first;
		if(vis[node]) continue;
		ans+=value,cntt++,vis[node]=1;
		for(int i=elast[node];~i;i=e[i].next){
			int j=e[i].v;
			if(!vis[j]){
				q.push({-e[i].w,j});
			}
		}
	}
	if(cntt!=n-1) return -1;
	return ans;
}
int main(){
	cin>>n>>m;
	memset(elast,-1,sizeof(elast)); 
	for(int i=0;i<m;i++){
		int x,y,z; 
		cin>>x>>y>>z;
		add(x,y,z);
		add(y,x,z);
	}
	int t=prim(1);
	if(t==-1) cout<<"orz";
	else cout<<t;
}
```



## 单源最短路径

模板： [P4779 【模板】单源最短路径（标准版）](https://www.luogu.com.cn/problem/P4779)

### Dijkstra算法

用于非负边权的单源最短路求解。

##### 流程

将结点分成两个集合：已确定最短路长度的点集（记为$S$集合）的和未确定最短路长度的点集（记为$T$集合）。一开始所有的点都属于$T$集合。

初始化 ，其他点的$dis[]$均为  $ +\infty$。

然后重复这些操作：

1. 从 集合中，选取一个最短路长度最小的结点，移到 集合中。
2. 对那些刚刚被加入 集合的结点的所有出边执行松弛操作。

直到 集合为空，算法结束。

##### 代码实现

**1.暴力**

 不使用任何数据结构进行维护，每次 2 操作执行完毕后，直接在 $T$ 集合中暴力寻找最短路长度最小的结点。 

**2.二叉堆优化** 

 每成功松弛一条边 $(u, v)$，就将 插入二叉堆中（如果 $v$ 已经在二叉堆中，直接修改相应元素的权值即可），1 操作直接取堆顶结点即可。共计m次二叉堆上的插入（修改）操作，n次删除堆顶操作。

```cpp
int dist[maxn], v[maxn];
priority_queue<pair<int, int> > q;
void dijkstra(){
	memset(dist, 0x3f, sizeof(dist));
	memset(v, 0, sizeof(v));
	dist[s] = 0;
	q.push(make_pair(0, s));
	while(!q.empty()){
		int x = q.top().second, d = q.top().first; q.pop();
		if(v[x]) continue;
		v[x] = 1;
		for(int i = head[x]; i; i = e[i].nxt){
			int y = e[i].v, z = e[i].w;
			if(dist[y] > dist[x] + z){
				dist[y] = dist[x] + z;
				q.push(make_pair(-dist[y], y));
			}
		}
	}
}
```

这里借鉴了李煜东（Rainbow）的《算法竞赛进阶指南》，通过存负值来保证小根堆性质，也可手写一个结构体

### SPFA 

$SPFA$ 是 $Bellman-Ford$ 算法的优先队列优化，其复杂度在一般情况下为 $ O (km)$ ,其中$k$通常是一个很小的常数，但在一些极端情况下其复杂度可退化至 $ O(nm)$,和暴力的BF一样。

所以，它死了。

但是还是挺好用的。

```cpp
void spfa(){
    vis[0] = 1;
	q.push(0);
	while (!q.empty()) {
	    int u = q.front(); q.pop(); vis[u] = 0;
	    if (tot[u] == n - 1) { cout << -1; return 0; }
	    tot[u]++;
	    for (int i = head[u]; i; i = e[i].nxt)
	        if (dis[e[i].v] < dis[u] + e[i].w) {
	            dis[e[i].v] = dis[u] + e[i].w;
	            if (!vis[e[i].v]) vis[e[i].v] = 1, q.push(e[i].v);
	        }
	}
}	
```



### Floyd



### 应用

#### 差分约束

模板：[P5960 【模板】差分约束算法](https://www.luogu.com.cn/problem/P5960)

可以看到每个不等式和最短路中 `dis[e[i].v] < dis[u] + e[i].w` 是很像的，所以我们可以建一张图，求从0到每个点的最短路（或最长路）就行了。

```cpp
#include<bits/stdc++.h>
#define maxn 10005
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
} 
struct edge{
	int v, w, nxt;
}e[maxn];
int head[maxn], cnt;
int n, m;
void add(int u, int v, int w){
	e[++cnt].v = v, e[cnt].w = w;
	e[cnt].nxt = head[u], head[u] = cnt;
}
int dis[maxn], vis[maxn], in[maxn];
bool spfa(){
	queue<int> q;
	memset(dis, -1, sizeof(dis));
	memset(vis, 0, sizeof(vis));
	memset(in, 0, sizeof(in));
	dis[0] = 0, vis[0] = 1, in[0] = 1;
	q.push(0);
	while(q.size()){
		int x = q.front(); q.pop();
		vis[x] = 0;
		for(int i = head[x]; i; i = e[i].nxt){
			int y = e[i].v, z = e[i].w;
			if(dis[y] < dis[x] + z){
				dis[y] = dis[x] + z;
				if(!vis[y]){
					q.push(y), vis[y] = 1;
					++in[y];
					if(in[y] > n + 1) return 1;
				}
			}
		}
	}
	return 0;
}
int main() {
	n = read(), m = read();
	for(int i = 1; i <= m; i++){
		int u = read(), v = read(), w = read();
		add(u, v, -w);
	}
	for(int i = 1; i <= n; i++) add(0, i, 0);
	if(spfa()) cout << "NO";
	else for(int i = 1; i <= n; i++) cout << dis[i] << " ";
}
```



#### 负环

模板：[P3385 【模板】负环](https://www.luogu.com.cn/problem/P3385) 

用 $SPFA$ 求负环，需要加一个 $in$ 数组来记录这条最短路上走的编书，由于 $1到x$ 最短路上的边数肯定是 $n-1$，所以一旦 $in[y] >= n$ 即存在环。

```cpp
#include<bits/stdc++.h>
#define maxn 50005
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
} 
struct edge{
	int v, w, nxt;
}e[maxn];
int head[maxn], cnt;
int n, m;
void add(int u, int v, int w){
	e[++cnt].v = v, e[cnt].w = w;
	e[cnt].nxt = head[u], head[u] = cnt;
}
void pre(){
	for(int i = 1; i <= cnt; i++) e[i].nxt = e[i].v = e[i].w = head[i] = 0;
	cnt = 0;
}
int dis[maxn], vis[maxn], in[maxn];
bool spfa(){
	queue<int> q;
	memset(dis, 0x3f, sizeof(dis));
	memset(vis, 0, sizeof(vis));
	memset(in, 0, sizeof(in));
	dis[1] = 0, vis[1] = 1;
	q.push(1);
	while(q.size()){
		int x = q.front(); q.pop();
		vis[x] = 0;
		for(int i = head[x]; i; i = e[i].nxt){
			int y = e[i].v, z = e[i].w;
			if(dis[y] > dis[x] + z){
				dis[y] = dis[x] + z;
				in[y] = in[x] + 1;
				if(in[y] >= n) return 1; 
				if(!vis[y]){
					q.push(y), vis[y] = 1;
					
				}
			}
		}
	}
	return 0;
}
int main() {
	int T = read();
	while(T--){
		n = read(), m = read();
		pre();
		for(int i = 1; i <= m; i++){
			int u = read(), v = read(), w = read();
			add(u, v, w);
			if(w >= 0) add(v, u, w);
		}
		if(spfa() == 1) cout << "YES" << endl;
		else cout << "NO" << endl;
	}
}
```

 

## 图的连通性相关

### 缩点

**模板**：[P3387 【模板】缩点](https://www.luogu.com.cn/problem/P3387) 

这道题如果没有环的话用拓扑排序就可以了，然而这道题是有环的。这些存在从 $x$ 到 $y$ 的路径，也存在从 $y$ 到 $x$ 的路径，这叫做强连通分量。

对于每个连通分量，我们可以把它缩成一个点，因为如果这个连通分量中有一个点可以经过，那么整个连通分量也可以经过，也就是说，选了这个连通分量中的一个点，其他点也选上才是最优的。

缩完点后就是一张新的图，我们在这张新图上拓扑即可。

下面是缩点和建图代码：

```cpp
void add(int u, int v){
	e[++cnt].v = v, e[cnt].u = u;
	e[cnt].nxt = head[u], head[u] = cnt;
}
int dfn[maxn], low[maxn];
int s[maxn];
int n, m;
int tme, top;
int p[maxn], sd[maxn], in[maxn], vis[maxn];
void tarjan(int x){
	low[x] = dfn[x] = ++tme;
	s[++top] = x, vis[x] = 1;
	for(int i = head[x]; i; i = e[i].nxt){
		int v = e[i].v;
		if(!dfn[v]){
			tarjan(v);
			low[x] = min(low[x], low[v]);
		}
		else if(vis[v]) low[x] = min(low[x], dfn[v]);
	}
	if(dfn[x] == low[x]){
		int y = 0;
		while(y = s[top--]){
			sd[y] = x;
			vis[y] = 0;
			if(x == y) break;
			p[x] += p[y];
		}
	}	
}
int d[maxn];

	cin >> n >> m;
	for(int i = 1; i <= n; i++) sd[i] = i;
	for(int i = 1; i <= n; i++) cin >> p[i];
	for(int i = 1; i <= m; i++){
		int u ,v ; cin >> u >> v;
		add(u, v);
	}
	for(int i = 1; i <= n; i++) if(!dfn[i]) tarjan(i);
	for(int i = 1; i <= m; i++){
		int x = sd[e[i].u], y = sd[e[i].v];
		if(x != y){
			e2[++cnt2].nxt = head2[x];
			e2[cnt2].v = y;
			e2[cnt2].u = x;
			head2[x] = cnt2;
			in[y] ++;
		}
	}


```



### 割点

[P3388 【模板】割点（割顶）](https://www.luogu.com.cn/problem/P3388) 

```cpp

```



## 二分图

### 二分图匹配——匈牙利算法

模板：[P3386 【模板】二分图最大匹配](https://www.luogu.com.cn/problem/P3386) 

```cpp
#include <bits/stdc++.h>
#define maxn 100000
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
}
struct edge{
	int u, v, nxt;
}e[maxn]; 
int head[maxn],head2[maxn], cnt, cnt2;
void add(int u, int v){
	e[++cnt].v = v, e[cnt].u = u;
	e[cnt].nxt = head[u], head[u] = cnt;
}
int vis[maxn], match[maxn];
bool dfs(int x){
	for(int i = head[x]; i; i = e[i].nxt){
		int y = e[i].v;
		if(!vis[y]){
			vis[y] = 1;
			if(!match[y] || dfs(match[y])){
				match[y] = x;
				return 1;
			}
		}
	}
	return false;
}
int n, m, c;
int sum = 0;
int main(){
	n = read(), m = read(), c = read();
	for(int i = 1; i <= c; i++){
		int u = read(), v = read();
		add(u, v + n);
	}
	for(int i = 1; i <= n; i++){
		memset(vis, 0, sizeof(vis));
		if(dfs(i)) ++sum;
	}
	cout << sum << endl;
	
}
```



## 网络流

### 网络最大流

**EK算法：** 

```cpp
#include <bits/stdc++.h>
#define int long long
#define maxn 10005
#define inf 1e9
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
}
struct edge{
	int v, w, nxt;
}e[maxn];
int head[maxn], cnt;
int n, m, s, t;
void add(int u, int v, int w){
	e[++cnt].v = v, e[cnt].w = w;
	e[cnt].nxt = head[u], head[u] = cnt;
}
queue<int> q;
int v[maxn];
int maxflow = 0;
int incf[maxn], pre[maxn];
bool bfs(){
	memset(v, 0, sizeof(v));
	while(q.size()) q.pop();
	q.push(s), v[s] = 1;
	incf[s] = inf;
	while(!q.empty()){
		int x = q.front();
		q.pop();
		for(int i = head[x]; i; i = e[i].nxt){
			if(e[i].w){
				int y = e[i].v;
				if(v[y]) continue;
				incf[y] = min(incf[x], e[i].w);
				pre[y] = i;
				q.push(y);
				v[y] = 1;
				if(y == t) return 1;
			}
		}
	}
	return 0;
}
void update(){
	int x = t;
	while(x != s){
		int i = pre[x];
		e[i].w -= incf[t];
		e[i ^ 1].w += incf[t];
		x = e[i ^ 1].v;
	}
	maxflow += incf[t];
}
signed main() {
	n = read(), m = read(), s = read(), t = read();
	cnt = 1;
	for(int i = 1; i <= m; i++){
		int u = read(), v = read(), w = read();
		add(u, v, w), add(v, u, 0);
	}
	while(bfs()) update();
	cout << maxflow;
}
```

**dinic算法：** 

```cpp
#include <bits/stdc++.h>
#define int long long
#define maxn 10005
#define inf 1e9
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
}
struct edge{
	int v, w, nxt;
}e[maxn];
int head[maxn], cnt;
int n, m, s, t;
void add(int u, int v, int w){
	e[++cnt].v = v, e[cnt].w = w;
	e[cnt].nxt = head[u], head[u] = cnt;
}
queue<int> q;
int d[maxn], now[maxn];
bool bfs(){
	memset(d, 0, sizeof(d));
	while(q.size()) q.pop();
	q.push(s);
	d[s] = 1;
	now[s] = head[s];
	while(!q.empty()){
		int x = q.front(); q.pop();
		for(int i = head[x]; i; i = e[i].nxt){
			if(e[i].w && !d[e[i].v]){
				q.push(e[i].v);
				now[e[i].v] = head[e[i].v];
				d[e[i].v] = d[x] + 1;
				if(e[i].v == t) return 1;
			}
		}
	}
	return 0;
}
int dinic(int x, int flow){
	if(x == t) return flow;
	int rest = flow, k, i;
	for(int i = now[x]; i && rest; i = e[i].nxt){
		now[x] = i;
		if(e[i].w && d[e[i].v] == d[x] + 1){
			k = dinic(e[i].v, min(rest, e[i].w));
			if(!k) d[e[i].v] = 0;
			e[i].w -= k;
			e[i ^ 1].w += k;
			rest -= k;
		}
	}
	return flow - rest;
}
int maxflow = 0;
signed main() {
	n = read(), m = read(), s = read(), t = read();
	cnt = 1;
	for(int i = 1; i <= m; i++){
		int u = read(), v = read(), w = read();
		add(u, v, w), add(v, u, 0);
	}
	int flow = 0;
	while(bfs()){
		while(flow = dinic(s, inf)) maxflow += flow;
	}
	cout << maxflow;
}
```



## 树上问题

### 树的直径

树的直径是指树上最远两点的距离。

可由两种方法求：

**1.树上DP**

**2.两遍dfs**

有一个结论：从树上任意一点出发，到达其所能到达的最远点，记作$num$,再从$num$出发到达其所能到达的最远点 $num2$ ,则 $num1$ 和 $num2$ 之间的距离就是树的直径。

通过这个结论，我们可以通过两遍dfs求直径。







# 动态规划

> 动态规划是一种通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。
>
> 由于动态规划并不是某种具体的算法，而是一种解决特定问题的方法，因此它会出现在各式各样的数据结构中，与之相关的题目种类也更为繁杂。
>
> 在 OI 中，计数等非最优化问题的递推解法也常被不规范地称作 DP   ——[oi.wiki](oi.wiki)

## 线性DP

### 背包问题

#### 01背包

模板：[P1048 [NOIP2005 普及组] 采药](https://www.luogu.com.cn/problem/P1048) 

```cpp
#include<bits/stdc++.h>
using namespace std;
int w[101],v[101],dp[1001];
int main(){
    int t,m;    
    cin >> t >> m;
    for(int i = 1; i <= m;i ++)
    	cin >> w[i] >> v[i];
    for(int i = 1; i <= m;i++) {
        for(int j = t; j >= 0;j--) {
            if(j >= w[i])
                dp[j] = max(dp[j-w[i]] + v[i], dp[j]);
        }
    }    
    cout<<dp[t];
}
```



#### 多重背包

转化为01背包求解

#### 完全背包

模板：[P1616 疯狂的采药](https://www.luogu.com.cn/problem/P1616) 

```cpp
#include<bits/stdc++.h>
using namespace std;
long long w[100001],c[100001];
long long dp[10000005];
int main(){
    int v,n;
    cin>>v>>n;
    for(int i=1;i<=n;i++)
        cin>>w[i]>>c[i];
    for(int i=1;i<=n;i++)
        for(int V=w[i];V<=v;V++)
            dp[V]=max(dp[V],dp[V-w[i]]+c[i]);
    cout<<dp[v];
}
```



#### 混合背包

模板：[P1833 樱花](https://www.luogu.com.cn/problem/P1833) 

朴素解法：对于每个物品，判断它属于哪种背包，然后分别用那种背包进行DP。

```cpp
#include<bits/stdc++.h>
using namespace std;
int dp[100005];
int v[100005], w[100005], p[100005], n;
int main(){
	int te, tt, t1, t2;
	char c;
	cin >> t1 >> c >> t2; te = t1*60 + t2;
	cin >> t1 >> c >> t2; tt = t1*60 + t2;
	int t = tt - te;
	cin >> n;
	for(int i = 1; i <= n; i++) cin >> w[i] >> v[i] >> p[i];
	for(int i = 1; i <= n; i++){
		if(p[i] == 0)
			for(int j = w[i]; j <= t; j++)
				dp[j] = max(dp[j], dp[j-w[i]] + v[i]);
		else{
			for(int j = 1; j <= p[i]; j++){
				for(int k = t; k >= w[i]; k--){
					dp[k] = max(dp[k], dp[k-w[i]] + v[i]);
				}
			}
		}
	}
	cout << dp[t];
}
```

得分：$80pts$ 原因：超时

于是考虑一些高效解法：**二进制拆分** 

做法：把每一个物品**根据2的多少次方拆分**，因为任何数都可以转化为二进制数；

**核心思想：把每一个物品拆成很多个，分别计算价值和所需时间，再转化为01背包求解；**

最后一点：**完全背包可以把他的空间记为999999**，不要太大，一般百万就足够了；

记得这时候数组要开大一点，因为是把一个物品拆成多个物品了

```cpp
#include<bits/stdc++.h>
using namespace std;
int dp[20005];
int v[20005], w[20005], p[20005], n;
int top;
int vo[100005], co[100005];
void binary_split(){	//二进制拆分
	for(int i = 1; i <= n; i++){
		int cf = 1;
		while(p[i] != 0){
			co[++top] = w[i] *cf;
			vo[top] = v[i] * cf;
			p[i] -= cf;
			cf *= 2;
			if(p[i] < cf){
				co[++top] = w[i] * p[i];
				vo[top] = v[i] * p[i];
				break;
			}
		}
	}
}
int main(){
	int te, tt, t1, t2;
	char c;
	cin >> t1 >> c >> t2; te = t1*60 + t2;
	cin >> t1 >> c >> t2; tt = t1*60 + t2;
	int t = tt - te;
	cin >> n;
	for(int i = 1; i <= n; i++){
		cin >> w[i] >> v[i] >> p[i];
		if(!p[i]) p[i] = 999999;
	}
	binary_split();
	for(int i = 1; i <= top; i++){
		for(int j = t; j >= co[i]; j--){
			dp[j] = max(dp[j], dp[j-co[i]] + vo[i]);
		}
	}
	
	cout << dp[t];
}
```



## 区间DP





## 树形DP

树形 DP，即在树上进行的 DP。由于树固有的递归性质，树形 DP 一般都是递归进行的。

树形 DP 的主要实现形式就是 $dfs$ 。

### 基本的状态转移方程

**选择节点类：** 

$dp[i][0]=dp[j][1]$

$dp[i][1]=max/min(dp[j][0],dp[j][1])$

**背包类：** 

$dp[v][k]=dp[u][k]+val$

$dp[u][k]=max(dp[u][k],dp[v][k−1])$

树形DP一般没有什么固定做法，一种题目有一种题目的做法。



### 例题





## 状态压缩DP

>  状压 DP 是动态规划的一种，通过将状态压缩为整数来达到优化转移的目的。——[oi.wiki](https://oi.wiki/dp) 

状压，即**状态压缩**的简称，是一种（**在数据范围较小的情况下**）将每个物品或者东西选与不选的**状态“压”成一个整数**的方法

通常我们采用**二进制状压法**，即对于一个我们“压”成的状态，**这个整数在二进制下中的1表示某个物品已选，而0代表某个物品未选，这样我们就可以通过枚举这些“压”成的整数来达到枚举所有的物品选与不选的总情况**，通常我们称这些情况叫做**子集**，对于这个状态整数，通常设为s

**（对于二进制状压）通过二进制下的位运算来达到判断某个物品选与不选的情况，再通过这个状态来进行一些其他的扩展，所以状压能简化我们对于问题的求解方式**

而状压 $dp$ 正是用到了这一点，通过一个状态来表示整体情况，对于这个情况进行一些最优化操作，最终达到求得全局最优解的目的

首先二进制状压通常要用到一些位运算：

![](D:\拾荒记\图片\状压DP_位运算.png)

------

通过一道[例题(P3959 [NOIP2017 提高组] 宝藏)](https://www.luogu.com.cn/problem/P3959)来了解状压DP：

首先我们注意到数据范围：$n <= 12$ ，这提醒我们：可以用状态压缩；

然后整理一下题意，其实就是找一个连边顺序使得所有点连通且代价最小，其中对于一个节点 $x$ **其可以连通的节点必然对应着两种状态：选或不选**，所以可以用状压来做。

设 $s$ 为已经打通的点所构成的集合，dp[i] [s] [deep] 表示当前 **i** 节点所对应的选点集合为**s**，深度为**deep**

可以写出状态转移方程：$ dp[j] [1<<(j-1) | s] [deep + 1] = dp[i] [s] [deep] + dis[i] * edge_{i , j}$

由于起点不确定，对于每个点都 $dfs$ 一遍即可。

代码中有详细解释：

```cpp
#include<bits/stdc++.h>
#define maxn 100005
#define int long long
const int mod = 1e9 + 7;
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
}
int dis[20], dp[15][1<<15][15];
int lim;
int e[20][20]; //用邻接矩阵来存图，可以快速找到从i到j的边权
int as = 1e9;	//把答案先初始成一个极大值
int n, m;
void dfs(int s, int sum, int deep){
	if(sum >= as) return;	//剪枝
	if(s == lim){	
        as = sum;	//已经把lim初始化成(1<<n) - 1, 也就是对应所有节点都加入集合的状态
        return;
    }
	for(int i = 1; i <= n; i++){
		if(!(1<<(i-1) & s)) continue;	//如果i不在s集合中就跳过，因为此时的i不属于这个状态
		for(int j = 1; j <= n; j++){
			if(!(1<<(j-1) & s) && e[i][j] < 1e9{	//j还没有被探索且i，j之间存在边
				if(dp[j][1<<(j-1) | s][deep + 1] <= sum + dis[i] * e[i][j]) continue;
                //如果此时的dp所存的最优解要更优就不需要进行状态转移
				dp[j][1<<(j-1) | s][deep + 1] = sum + dis[i] * e[i][j];//状态转移方程
				dis[j] = dis[i] + 1;//路径增加
				dfs(1<<(j-1) | s, dp[j][1<<(j-1) | s][deep + 1], deep + 1);
                //在此状态基础上继续搜
			}
		}
	}
}
signed main(){
	n = read(), m = read();
	lim = (1<<n) - 1;
	memset(e, 0x3f, sizeof(e));
	for(int i = 1; i <= m; i++){
		int u = read(), v = read(), w = read();
		e[u][v] = e[v][u] = min(e[u][v], w);//处理重边，因为n只有12而m很大
	}
	for(int i = 1; i <= n; i++){	//每个点都遍历一次
		memset(dis, 0, sizeof(dis));
		memset(dp, 0x3f, sizeof(dp));
		dis[i] = 1;
		dfs(1<<(i-1), 0, 0);
	}
	cout << as;
}
```





## 数位DP

首先来关注一些**概念**：

>  **数位**：把一个数字按照个、十、百、千等等一位一位地拆开，**关注它每一位上的数字**。如果拆的是十进制数，那么每一位数字都是 0~9，其他进制可类比十进制。 

> **数位 DP**：用来解决一类特定问题，这种问题比较好辨认，一般具有这几个**特征**：
>
> 1. 要求统计满足一定条件的数的数量（即，最终目的为计数）；
> 2. 这些条件经过转化后可以使用**「数位」**的思想去理解和判断；
> 3. 输入会提供一个数字区间（有时也只提供上界）来作为统计的限制；
> 4. 上界很大（比如 $10^18$ ），暴力枚举验证会超时。

以上概念均引自[oi.wiki](oi.wiki)

### 基本原理

考虑人类计数的方式，最朴素的计数就是从小到大开始依次加一。但我们发现对于位数比较多的数，这样的过程中有许多重复的部分。例如，从 7000 数到 7999、从 8000 数到 8999、和从 9000 数到 9999 的过程非常相似，它们都是后三位从 000 变到 999，不一样的地方只有千位这一位，所以我们可以把这些过程归并起来，将这些过程中产生的计数答案也都存在一个通用的数组里。此数组根据题目具体要求设置状态，用递推或 DP 的方式进行状态转移。

数位 DP 中通常会利用常规计数问题技巧，比如把一个区间内的答案拆成两部分相减

### 基本代码

来看一道**模板题**：[P2602 [ZJOI2010] 数字计数](https://www.luogu.com.cn/problem/P2602)

这道题明显符合 数位DP 的特征，所以用数位的方法做：求出 $[0, a-1]$ 的统计与 $[0, b]$ 的统计然后相减；

首先我们需要一个辅助数组 $f$ ， $f[i]$ 代表在有i位数字的情况下，每个数字有多少个。如果不考虑前导0，你会发现对于每一个数，它的数量都是相等的，也就是 $f[i]=f[i-1]*10+10^(i-1)$ ; 

怎么知道我们想要的答案呢？

设我们要推得数是 ABCD， 我们先统计最高位： 鉴于我们其实已经求出了0~9,0~99,0~999……上所有数字个数（f[i],且没有考虑前导0）我们何不把这个A000看成0000~1000~2000...A000对于不考虑首位每一个式子的数字的出现个数为 A*f[3]。加上首位出现也就是小于A每一个数都出现了 10^3 次，再加上，我们就把A000处理完了。

但是首位处理还不止如此，要注意后面BCD的时候A还会再出现，所以次数加上 BCD+ 1；

然后是处理前导0：前导0情况一定是 0001、0002、0003……0999，0出现次数是 $10^2$， 所以0再减去 $10^2$就行了。

其他位也是一样的，递推即可。

```cpp
#include<bits/stdc++.h>
#define maxn 100005
#define int long long //注意范围开longlong
using namespace std;
inline int read(){
    int x = 0 , f = 1 ; char c = getchar() ;
    while( c < '0' || c > '9' ) { if( c == '-' ) f = -1 ; c = getchar() ; } 
    while( c >= '0' && c <= '9' ) { x = x * 10 + c - '0' ; c = getchar() ; } 
    return x * f ;
}
int a, b;
int f[100];
int cnta[100], cntb[100], ten[100];
void solve(int x, int *a){
	int num[100] = {0};
	int len = 0;
	while(x){
		num[++len] = x%10;		//将数展开
		x = x/10;
	}
	for(int i = len; i >= 1; i--){		//一位一位递推
		for(int j = 0; j <= 9; j++)		
			a[j] += f[i-1] * num[i];
		for(int j = 0; j < num[i]; j++)	
			a[j] += ten[i-1];
		int num2 = 0;
		for(int j = i-1; j >= 1; j--){
			num2 = num2*10 + num[j];
		}
		a[num[i]] += num2 + 1;
		a[0] -= ten[i-1];
	}
}
signed main(){
	a = read(), b = read();
	ten[0] = 1;
	for(int i = 1; i <= 15; i++){
		f[i] = f[i-1]*10 + ten[i-1];	//预处理
		ten[i] = 10*ten[i-1];
	} 
	solve(a-1, cnta), solve(b, cntb);
	for(int i = 0; i <= 9; i++) cout << cntb[i] - cnta[i] << " ";
}
```

 



# 杂项

## C++高精

`__int128`

## 分数

```cpp
struct node{
    ll p, q;
    node(){
        p = 0, q = 1;
    }
    node operator *(const ll &rhs) const {
        node res;
        res.p = p, res.q = q * rhs;
        ll g = __gcd(res.p, res.q);
        res.p /= g, res.q /= g;
        return res;
    }
    node operator +(const node &rhs) const {
        node res;
        res.q = lcm(q, rhs.q);
        res.p += p * (res.q / q);
        res.p += rhs.p * (res.q / rhs.q);
        ll g = __gcd(res.p, res.q);
        res.p /= g, res.q /= g;
        return res;
    }
}ans[maxn];
void print(int n) {
    if(n > 9) print(n / 10);
    putchar(n % 10 + 48);
}


print(ans[i].p);
cout << " / ";
print(ans[i].q);
cout << endl;
```
