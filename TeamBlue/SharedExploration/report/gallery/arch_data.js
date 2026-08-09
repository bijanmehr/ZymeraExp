/* Architecture reference — the REAL learning stack (ctde_v0/ + zymera_lab/zymera/), rendered as
   block-and-flow diagrams by pages/architecture.html. Each entry: {graph} (nodes on a col/row grid +
   edges) and {detail} (machinery prose + shapes + citations). Param counts derived from layer shapes. */
window.ARCH = [

/* ============================ FLOW ============================ */
{id:'system', name:'System flow (end to end)', group:'Flow', badge:'map',
 tagline:'From the world to an action and back — decentralised actors, one central critic, per step.',
 graph:{ nodes:[
   {id:'env',   col:1,row:0,kind:'env',  label:'Environment', sub:'grid world · walls'},
   {id:'obs',   col:0,row:1,kind:'input',label:'per-agent obs', sub:'(N,5,H,W)'},
   {id:'comm',  col:1,row:1,kind:'data', label:'comm graph Aₜ', sub:'range-limited'},
   {id:'cobs',  col:2,row:1,kind:'input',label:'central obs', sub:'(3,H,W)', op:'god-view'},
   {id:'bb',    col:0,row:2,kind:'conv', label:'actor backbone', op:'CNN→GAP→GNN', sub:'z (N,64)'},
   {id:'crit',  col:2,row:2,kind:'conv', label:'central critic', sub:'V (1)', op:'train-only'},
   {id:'goal',  col:0,row:3,kind:'head', label:'goal head', sub:'(N,9)'},
   {id:'role',  col:1,row:3,kind:'head', label:'role head', sub:'(N,2)'},
   {id:'ctrl',  col:0,row:4,kind:'ctrl', label:'L1 controller', op:'greedy / relay'},
   {id:'act',   col:1,row:4,kind:'env',  label:'env action', sub:'(N,) ∈ 5 moves'}
 ], edges:[
   {from:'env',to:'obs',label:'sense'},{from:'env',to:'comm'},{from:'env',to:'cobs'},
   {from:'obs',to:'bb'},{from:'comm',to:'bb',label:'adjacency'},{from:'cobs',to:'crit'},
   {from:'bb',to:'goal'},{from:'bb',to:'role'},
   {from:'goal',to:'ctrl',label:'sample'},{from:'role',to:'ctrl',label:'role'},
   {from:'ctrl',to:'act'},{from:'act',to:'env',kind:'feedback',route:'margin',label:'step t→t+1'}
 ]},
 detail:`
 <p>Every step: the world emits each agent a local <b>belief</b> observation and a range-limited
 <b>comm graph</b>; the decentralised actor turns that into a <b>goal + role</b>; a <em>scripted</em>
 controller turns those into one of five moves; the moves advance the world. A separate <b>central
 critic</b> sees a god-view — used only to train, never at execution. Interaction touches the mission
 <b>twice</b>: it shapes each agent's input (the GNN adjacency) and how local actions compose into the
 next state.</p>
 <p class="cite">ppo.py (rollout) · nets.py · controller.py · zymera env.py</p>`},

{id:'overview', name:'Learned vs scripted', group:'Flow', badge:'map',
 tagline:'What carries gradients, what is hand-coded, and why scale-invariance holds.',
 detail:`
 <h4>Learned (gradient or evolution)</h4>
 <p>goal head · role head · skill selector · FrontierAttn · compass · diversity residual · GRU · the central
 scalar critic. The λ̂₂ head is learned but <em>diagnostic</em> (supervised against an oracle), not the enforcement signal.</p>
 <h4>Scripted / non-gradient</h4>
 <p>the L1 controller (greedy-to-goal + relay moves) · the action-mask guardrail · every reward-term
 formula · the frontier / compass / congestion <em>features</em> · the connectivity dual multipliers
 (running λ / PID, not gradient-learned).</p>
 <div class="note"><b>Scale-invariance is load-bearing.</b> Global-average-pool (not flatten), normalised
 GNN aggregators, fixed i/N identity codes and fraction-valued features make every parameter <b>shape</b>
 depend only on <code>channels / width / depth / mp_rounds / K / n_roles</code> — never on grid size or
 agent count. That is what lets a 16²/4 checkpoint warm-start a 32²/10 run.</div>
 <p class="cite">nets.py · ppo.py · controller.py · checkpoint.py</p>`},

/* ============================ PERCEPTION ============================ */
{id:'obs', name:'Observation — belief build', group:'Perception', badge:'scripted',
 tagline:'Agents act on a gossiped belief, never ground truth; the critic gets the god-view.',
 graph:{ nodes:[
   {id:'world',col:0,row:0,kind:'env', label:'world state', sub:'walls · coverage', op:'hidden'},
   {id:'sense',col:0,row:1,kind:'data',label:'local sense', op:'Chebyshev sense_r'},
   {id:'goss', col:0,row:2,kind:'data',label:'gossip merge', op:'over comm graph'},
   {id:'bel',  col:0,row:3,kind:'data',label:'post-gossip belief', sub:'per agent'},
   {id:'plan', col:1,row:3,kind:'input',label:'obs planes', sub:'(N,5,H,W)'},
   {id:'cobs', col:1,row:1,kind:'input',label:'central obs', sub:'(3,H,W)', op:'god-view (train)'}
 ], edges:[
   {from:'world',to:'sense'},{from:'sense',to:'goss'},{from:'goss',to:'bel'},
   {from:'bel',to:'plan',label:'5 channels'},{from:'world',to:'cobs',label:'ground truth'}
 ]},
 detail:`
 <p>Default <b>C = 5</b> per-agent planes <code>(N,5,H,W)</code>; <code>+occ_frontier</code> with
 <code>--sense-free</code>, <code>+boundary</code> with <code>--boundary</code>. No learnable params — pure functions of the belief.</p>
 <table class="spec">
  <tr><th>#</th><th>channel</th><th>encodes</th></tr>
  <tr><td>0</td><td><code>known</code></td><td>belief map (1 = known/covered)</td></tr>
  <tr><td>1</td><td><code>own_pos</code></td><td>one-hot of the agent's own cell</td></tr>
  <tr><td>2</td><td><code>known_walls</code></td><td>walls the agent has learned</td></tr>
  <tr><td>3</td><td><code>neighbors</code></td><td>one-hots of in-range teammates</td></tr>
  <tr><td>4</td><td><code>local_frontier</code></td><td>unknown within Chebyshev sense_r</td></tr>
 </table>
 <p>The central-critic view (<code>Cg = 3</code>: team_explored · all_pos · walls) is ground truth,
 reserved for training. Role is a head <em>output</em>, not a channel; the comm graph enters only as the
 <code>neighbors</code> plane and the GNN adjacency.</p>
 <p class="cite">zymera obs.py (CHANNEL_FNS) · env.py:518-534 · config.py:46-54</p>`},

/* ============================ POLICY ============================ */
{id:'actor', name:'Actor — LPAC backbone + heads', group:'Policy', badge:'learned',
 tagline:'Per-agent CNN → global-average-pool → GNN → belief z → four linear heads.',
 graph:{ nodes:[
   {id:'obs', col:1,row:0,kind:'input',label:'obs', sub:'(N,5,H,W)'},
   {id:'c1',  col:1,row:1,kind:'conv', label:'Conv 3×3 5→64', op:'ReLU', sub:'(N,64,H,W)'},
   {id:'c2',  col:1,row:2,kind:'conv', label:'Conv 3×3 64→64', op:'ReLU', sub:'(N,64,H,W)'},
   {id:'gap', col:1,row:3,kind:'pool', label:'Global Avg Pool', sub:'(N,64)'},
   {id:'gnn', col:1,row:4,kind:'gnn',  label:'GNN ×2 (comm graph)', sub:'z (N,64)'},
   {id:'goal',col:0,row:5,kind:'head', label:'goal 64→9', sub:'(N,9)'},
   {id:'role',col:1,row:5,kind:'head', label:'role 64→2', sub:'(N,2)'},
   {id:'aux', col:2,row:5,kind:'head', label:'aux 64→1', sub:'λ̂₂'},
   {id:'val', col:3,row:5,kind:'head', label:'value 64→1', sub:'baseline'}
 ], edges:[
   {from:'obs',to:'c1'},{from:'c1',to:'c2'},{from:'c2',to:'gap'},{from:'gap',to:'gnn'},
   {from:'gnn',to:'goal'},{from:'gnn',to:'role'},{from:'gnn',to:'aux'},{from:'gnn',to:'val'}
 ]},
 detail:`
 <p>Defaults: <b>width 64, depth 2, mp_rounds 2, agg = max</b>. The CNN is per-agent; the GNN is the only
 block that mixes across agents. The <b>goal head</b> is the PPO policy — logits over <b>K=9 relative
 compass waypoints</b>; the net never emits a raw move.</p>
 <h4>Layer-by-layer (model.summary style)</h4>
 <p style="font-size:12px;color:#7a8494;margin:.2em 0">Shapes symbolic — <b>N</b> agents, <b>H×W</b> grid. Convs are 3×3, stride 1, same-padding (spatial dims preserved); param counts are fixed regardless of H, W, N.</p>
 <table class="summ">
  <tr><th>#</th><th>layer</th><th>output</th><th>params</th><th>what it means</th></tr>
  <tr><td>0</td><td class="op">Input · obs planes</td><td class="shape">(N,5,H,W)</td><td class="p">—</td><td>5 belief planes per agent</td></tr>
  <tr><td>1</td><td class="op">Conv2d 3×3 5→64 + ReLU</td><td class="shape">(N,64,H,W)</td><td class="p">2,944</td><td>local detectors — frontier edges, walls, neighbours (3×3 field)</td></tr>
  <tr><td>2</td><td class="op">Conv2d 3×3 64→64 + ReLU</td><td class="shape">(N,64,H,W)</td><td class="p">36,928</td><td>compose into richer per-cell features (5×5 field)</td></tr>
  <tr><td>3</td><td class="op">GlobalAvgPool(H,W)</td><td class="shape">(N,64)</td><td class="p">0</td><td>collapse the map → one <b>size-invariant</b> vector per agent</td></tr>
  <tr><td>4</td><td class="op">LayerNorm</td><td class="shape">(N,64)</td><td class="p">128</td><td>stabilise embedding scale</td></tr>
  <tr><td>5</td><td class="op">GNN MPLayer ×2 (agg=max)</td><td class="shape">(N,64) = z</td><td class="p">24,832</td><td>fuse neighbour beliefs over 2 comm-hops → the shared picture</td></tr>
  <tr><td>6a</td><td class="op">goal_head Linear 64→9</td><td class="shape">(N,9)</td><td class="p">585</td><td>PPO policy — logits over 9 compass waypoints</td></tr>
  <tr><td>6b</td><td class="op">role_head Linear 64→2</td><td class="shape">(N,2)</td><td class="p">130</td><td>explorer vs relay</td></tr>
  <tr><td>6c</td><td class="op">aux_head Linear 64→1</td><td class="shape">(N,1)</td><td class="p">65</td><td>local λ̂₂ estimate (supervised)</td></tr>
  <tr><td>6d</td><td class="op">value_head Linear 64→1</td><td class="shape">(N,1)</td><td class="p">65</td><td>per-agent baseline</td></tr>
  <tr class="tot"><td></td><td>active path</td><td class="shape"></td><td class="p">~65.7k</td><td>+ ~90k always-built (GRU 24.8k · frontier-attn · compass · selector · flock) in the param tree</td></tr>
 </table>
 <p>Always-built-but-gated (present so an all-default net is byte-identical pre/post-feature):
 <code>frontier_attn · compass · goal_residual · gru · selector_head · flock</code>.
 <b>Full tree ≈ 150–200k</b>; every shape depends only on channels/width/depth/K — never grid or agent count.</p>
 <p class="cite">nets.py Actor 667-989, Backbone 248-292</p>`},

{id:'gnn', name:'GNN message-passing round', group:'Policy', badge:'learned',
 tagline:'For each agent: message every neighbour, aggregate, update — the cross-agent mixing.',
 graph:{ nodes:[
   {id:'hi', col:0,row:1,kind:'data',label:'hᵢ', sub:'(64)'},
   {id:'hj', col:2,row:0,kind:'data',label:'h_j', sub:'j ∈ N(i)'},
   {id:'msg',col:2,row:1,kind:'gnn', label:'msg: Lin 64→64'},
   {id:'agg',col:2,row:2,kind:'gnn', label:'aggregate', op:'max, normalised'},
   {id:'cat',col:1,row:2,kind:'data',label:'[hᵢ ‖ mᵢ]', sub:'(128)'},
   {id:'upd',col:1,row:3,kind:'gnn', label:'upd: Lin 128→64', op:'ReLU · 1 of 2 rounds'},
   {id:'ho', col:1,row:4,kind:'data',label:'hᵢ′', sub:'(64)'}
 ], edges:[
   {from:'hj',to:'msg',label:'each j'},{from:'msg',to:'agg'},
   {from:'hi',to:'cat'},{from:'agg',to:'cat',label:'mᵢ'},
   {from:'cat',to:'upd'},{from:'upd',to:'ho'}
 ]},
 detail:`
 <p>Two rounds. Each round, every agent <b>i</b> forms a message from each in-range neighbour, <b>aggregates</b>
 them (max, degree-normalised — never a raw sum, so the result is invariant to how many neighbours there are),
 concatenates with its own state, and <b>updates</b>. After 2 rounds an agent's belief <code>z</code> has mixed
 information up to 2 comm-hops away.</p>
 <h4>Inside one message-passing layer</h4>
 <table class="summ">
  <tr><th>step</th><th>op</th><th>output</th><th>params</th><th>meaning</th></tr>
  <tr><td>in</td><td class="op">node states H, adjacency A</td><td class="shape">(N,64) · (N,N)</td><td class="p">—</td><td>each agent's belief + who is in range</td></tr>
  <tr><td>1</td><td class="op">msg = Linear 64→64 (per node)</td><td class="shape">(N,64)</td><td class="p">4,160</td><td>what each agent tells its neighbours</td></tr>
  <tr><td>2</td><td class="op">aggregate max over N(i)</td><td class="shape">(N,64)</td><td class="p">0</td><td>pool neighbours → mᵢ (invariant to how many)</td></tr>
  <tr><td>3</td><td class="op">concat [hᵢ ‖ mᵢ]</td><td class="shape">(N,128)</td><td class="p">0</td><td>own state + neighbourhood</td></tr>
  <tr><td>4</td><td class="op">upd = Linear 128→64 + ReLU</td><td class="shape">(N,64) = hᵢ′</td><td class="p">8,256</td><td>revised belief</td></tr>
  <tr class="sub"><td></td><td>q,k Linear 64→64 (multihead only)</td><td class="shape"></td><td class="p">8,320</td><td>built; inactive under agg=max</td></tr>
  <tr class="tot"><td></td><td>per layer (max) · ×2 rounds</td><td class="shape"></td><td class="p">12,416 · 24,832</td><td>after 2 rounds z carries 2-hop info</td></tr>
 </table>
 <p><b>Aggregators:</b> <code>mean</code> (degree-normalised) · <code>max</code> (default) · <code>multihead</code>
 (softmax attention over neighbours, 4 heads). All size-invariant — the reason warm-start transfers across team size.</p>
 <p class="cite">nets.py MPLayer 119-230</p>`},

{id:'frontier', name:'FrontierAttn explorer tool', group:'Policy', badge:'learned',
 tagline:'Cross-attention over compass sectors biases the goal toward open ground.',
 graph:{ nodes:[
   {id:'z',    col:0,row:0,kind:'data',label:'belief z', sub:'(64)'},
   {id:'sect', col:2,row:0,kind:'data',label:'K sectors', sub:'frontier feat (K,2)'},
   {id:'q',    col:0,row:1,kind:'attn',label:'query 64→32'},
   {id:'k',    col:2,row:1,kind:'attn',label:'keys 2→32'},
   {id:'sc',   col:1,row:2,kind:'attn',label:'score q·kₖ/√d'},
   {id:'sm',   col:1,row:3,kind:'attn',label:'softmax over K'},
   {id:'ws',   col:1,row:4,kind:'attn',label:'K·attnₖ·fracₖ'},
   {id:'gate', col:0,row:4,kind:'head',label:'gate α', op:'softplus ≥ 0'},
   {id:'out',  col:1,row:5,kind:'head',label:'goal += α·frontier', sub:'(N,9)'}
 ], edges:[
   {from:'z',to:'q'},{from:'sect',to:'k'},{from:'q',to:'sc'},{from:'k',to:'sc'},
   {from:'sc',to:'sm'},{from:'sm',to:'ws'},{from:'ws',to:'out'},{from:'gate',to:'out'}
 ]},
 detail:`
 <p>The "disperse" skill (on with <code>--explorer-tool frontier_attn</code>). Attends over K compass
 <b>sectors</b> (not raw cells), then adds a frontier-seeking term into the goal logits — still sampled, never argmax.</p>
 <h4>Inside the attention</h4>
 <table class="summ">
  <tr><th>step</th><th>op</th><th>output</th><th>params</th><th>meaning</th></tr>
  <tr><td>in</td><td class="op">belief z ; sector feats F</td><td class="shape">(N,64) · (N,K,2)</td><td class="p">—</td><td>K compass sectors, each [frontier fraction, density]</td></tr>
  <tr><td>1</td><td class="op">query = Linear 64→32 (z)</td><td class="shape">(N,32)</td><td class="p">2,080</td><td>“what frontier am I seeking”</td></tr>
  <tr><td>2</td><td class="op">keys = Linear 2→32 (F)</td><td class="shape">(N,K,32)</td><td class="p">96</td><td>describe each sector</td></tr>
  <tr><td>3</td><td class="op">scores = q·kₖ / √32</td><td class="shape">(N,K)</td><td class="p">0</td><td>match query to sectors</td></tr>
  <tr><td>4</td><td class="op">softmax over K</td><td class="shape">(N,K)</td><td class="p">0</td><td>attention weight per sector</td></tr>
  <tr><td>5</td><td class="op">K · attnₖ · frontier_fracₖ</td><td class="shape">(N,K)</td><td class="p">0</td><td>frontier-weighted direction scores</td></tr>
  <tr><td>6</td><td class="op">gate α = softplus(logα)</td><td class="shape">scalar</td><td class="p">1</td><td>how hard to steer (learned)</td></tr>
  <tr><td>7</td><td class="op">goal_logits += α · frontier</td><td class="shape">(N,9)</td><td class="p">0</td><td>bias the policy toward open ground</td></tr>
  <tr class="tot"><td></td><td>total (q, k, α learned)</td><td class="shape"></td><td class="p">~2,177</td><td>frontier-positive even at init</td></tr>
 </table>
 <p class="cite">nets.py FrontierAttn 380-451</p>`},

{id:'action', name:'Hierarchical action (goal → move)', group:'Policy', badge:'mixed',
 tagline:'The learned goal head picks a waypoint; a scripted controller descends it to one of 5 moves.',
 graph:{ nodes:[
   {id:'z',   col:1,row:0,kind:'data',label:'belief z', sub:'(64)'},
   {id:'gl',  col:1,row:1,kind:'head',label:'goal_head 64→9'},
   {id:'st',  col:2,row:1,kind:'data',label:'compass stencil', op:'×stride 3'},
   {id:'sp',  col:1,row:2,kind:'head',label:'sample ~ softmax'},
   {id:'wp',  col:1,row:3,kind:'data',label:'waypoint (r,c)'},
   {id:'role',col:0,row:3,kind:'head',label:'role gate', sub:'expl / relay'},
   {id:'ct',  col:1,row:4,kind:'ctrl',label:'L1 controller', op:'greedy-to-goal / soft-deg climb'},
   {id:'mk',  col:2,row:4,kind:'ctrl',label:'conn guardrail', op:'optional'},
   {id:'mv',  col:1,row:5,kind:'env', label:'move ∈ {STAY,↑,↓,←,→}'}
 ], edges:[
   {from:'z',to:'gl'},{from:'st',to:'gl',label:'K=9'},{from:'gl',to:'sp'},{from:'sp',to:'wp'},
   {from:'wp',to:'ct'},{from:'role',to:'ct',label:'role'},{from:'mk',to:'ct',kind:'soft',label:'mask'},
   {from:'ct',to:'mv'}
 ]},
 detail:`
 <p>The stencil is <code>[here, N, E, S, W, NE, SE, SW, NW] × stride 3</code>. PPO samples a goal from the
 9 logits; the controller then walks one env step toward it (explorer) or climbs local soft-degree (relay).
 So <b>the decision is learned; the motion is scripted</b> — this is why the policy is size-invariant (a
 waypoint means the same thing on any grid) and why the whole thing survives warm-start.</p>
 <div class="note">Research runs use a <b>soft</b> connectivity signal, not the hard <code>action_mask</code>
 guardrail — an un-breakable graph would kill the resilience study.</div>
 <p class="cite">nets.py goal head · controller.py:38-233</p>`},

/* ============================ CRITIC ============================ */
{id:'critic', name:'Critic (central value)', group:'Critic', badge:'learned',
 tagline:'One scalar team value from the god-view — the only critic that trains the policy.',
 graph:{ nodes:[
   {id:'cobs',col:0,row:0,kind:'input',label:'central obs', sub:'(3,H,W)', op:'god-view'},
   {id:'cnn', col:0,row:1,kind:'conv', label:'Conv ×2 + GAP', op:'no GNN'},
   {id:'v',   col:0,row:2,kind:'head', label:'value_head → V', sub:'(1)', op:'trained'},
   {id:'gae', col:1,row:2,kind:'data', label:'GAE advantage', op:'→ PPO'}
 ], edges:[
   {from:'cobs',to:'cnn'},{from:'cnn',to:'v'},{from:'v',to:'gae'}
 ]},
 detail:`
 <p>CTDE: decentralised actors, one <b>centralised</b> critic that sees a god-view (training only). It outputs a
 single scalar team value that sources GAE + the value loss (<code>critic_mode="central"</code>, default).
 Same CNN→GAP as the actor backbone but <b>no GNN</b> and no per-agent heads (~39k).</p>
 <h4>Layer summary</h4>
 <table class="summ">
  <tr><th>#</th><th>layer</th><th>output</th><th>params</th><th>meaning</th></tr>
  <tr><td>0</td><td class="op">central obs (god-view)</td><td class="shape">(3,H,W)</td><td class="p">—</td><td>team_explored · all_pos · walls</td></tr>
  <tr><td>1</td><td class="op">Conv 3→64, 64→64 + GAP</td><td class="shape">(64)</td><td class="p">~39k</td><td>global scene summary (no GNN)</td></tr>
  <tr><td>2</td><td class="op">value_head Linear 64→1</td><td class="shape">(1)</td><td class="p">65</td><td>scalar team value V → GAE</td></tr>
 </table>
 <div class="note"><b>MAAC / COMA is in the code but NOT used.</b> An attention-critic (<code>AttnCritic</code>)
 + COMA counterfactual were built (task #68) as a per-agent contribution probe, but they <b>never enter
 training</b> and we <b>shelved</b> them: COMA's learned credit is too noisy for sparse coverage
 (credit↔truth r≈0.13), so the per-agent contribution / resilience measurement is moving to <b>exact,
 closed-form difference rewards</b> instead. Documented only so a reader doesn't wire them into the loss.</div>
 <p class="cite">trained: nets.py Critic 1062-1083 · ppo.py:825 &nbsp;|&nbsp; shelved: AttnCritic 1091-1170 · contribution.py</p>`},

/* ============================ COORDINATION ============================ */
{id:'role', name:'Role-picker (explorer / relay)', group:'Coordination', badge:'mixed',
 tagline:'Role choice is a learned PPO action; each role’s behaviour is scripted.',
 graph:{ nodes:[
   {id:'z',   col:1,row:0,kind:'data',label:'belief z', sub:'(64)'},
   {id:'rh',  col:1,row:1,kind:'head',label:'role_head 64→2'},
   {id:'sp',  col:1,row:2,kind:'head',label:'sample role', op:'PPO action'},
   {id:'ex',  col:0,row:3,kind:'ctrl',label:'explorer', op:'greedy-to-goal'},
   {id:'re',  col:2,row:3,kind:'ctrl',label:'relay', op:'soft-degree climb / hold'}
 ], edges:[
   {from:'z',to:'rh'},{from:'rh',to:'sp'},
   {from:'sp',to:'ex',label:'if explorer'},{from:'sp',to:'re',label:'if relay'}
 ]},
 detail:`
 <p><code>role_head: Linear(64→2)</code> (~130 params), sampled as a categorical PPO action with its entropy in
 the loss. On with <code>--role-picker expl_relay</code> (default off → all-explorer). The team's labour
 <b>division</b> is learned; each role's <b>execution</b> is heuristic. Superseded by the 3-skill
 <b>selector</b> {disperse, flock, hold} when <code>--selector on</code>.</p>
 <p class="cite">nets.py:780 · ppo.py:239-246 · controller.py:162-233</p>`},

{id:'connectivity', name:'Connectivity — the dual loop', group:'Coordination', badge:'scripted',
 tagline:'Measure the graph, form a violation, adapt a multiplier, penalise the reward — a control loop.',
 graph:{ nodes:[
   {id:'pol', col:0,row:1,kind:'head',label:'policy step'},
   {id:'meas',col:1,row:0,kind:'data',label:'measure λ₂', op:'or soft-degree'},
   {id:'tau', col:2,row:0,kind:'data',label:'floor τ', op:'target'},
   {id:'vio', col:1,row:1,kind:'data',label:'v = relu(τ − λ₂)'},
   {id:'dual',col:1,row:2,kind:'ctrl',label:'λ ← λ + lr·v', op:'or PID'},
   {id:'pen', col:1,row:3,kind:'loss',label:'− λ·shortfall'},
   {id:'rew', col:0,row:2,kind:'data',label:'team reward'}
 ], edges:[
   {from:'pol',to:'meas',label:'positions'},{from:'tau',to:'vio'},{from:'meas',to:'vio'},
   {from:'vio',to:'dual'},{from:'dual',to:'pen'},{from:'pen',to:'rew'},
   {from:'rew',to:'pol',kind:'feedback',label:'next step'}
 ]},
 detail:`
 <p>None of these add learnable parameters — the multiplier is a <b>running dual</b>, not a gradient variable.</p>
 <table class="spec">
  <tr><th>mechanism</th><th>what it is</th></tr>
  <tr><td><code>action_mask</code></td><td>hard guardrail (code default) — forbid goals that drop true λ₂ below a floor</td></tr>
  <tr><td><code>soft_lambda</code></td><td>fixed penalty <code>−penalty·shortfall</code> (<code>--soft-lambda-penalty</code>, the cov↔conn dial)</td></tr>
  <tr><td><code>lagrangian</code></td><td>running dual, ascended on the violation (<code>lambda_lr 0.05</code>)</td></tr>
  <tr><td><code>pid_lagrangian</code></td><td>λ from a PID controller (kp 1 · ki .01 · kd .1)</td></tr>
 </table>
 <p><b>conn-signal:</b> <code>global_lambda2</code> (one scalar, broadcast to all) vs <code>local_edge_margin</code>
 (per-agent soft-degree shortfall). Research runs use soft / lagrangian, not the hard mask.</p>
 <p class="cite">config.py:194-239 · env_utils.py:106-201 · ppo.py:268-277</p>`},

/* ============================ REWARD ============================ */
{id:'reward', name:'Reward — why each term', group:'Reward', badge:'scripted',
 tagline:'A reward-agnostic engine, weighted by the experiment: an objective (cover · connect · don’t crowd) plus small shaping.',
 graph:{ nodes:[
   {id:'cov', col:0,row:0,kind:'data',label:'coverage', op:'×1 · objective'},
   {id:'conn',col:0,row:1,kind:'data',label:'connectivity', op:'×2 · objective'},
   {id:'coll',col:0,row:2,kind:'data',label:'collision', op:'×−4 · guardrail'},
   {id:'anti',col:0,row:3,kind:'data',label:'anti-overlap', op:'×1 · shaping'},
   {id:'info',col:0,row:4,kind:'data',label:'info-gain', op:'×.1 · shaping'},
   {id:'cong',col:0,row:5,kind:'data',label:'congestion', op:'×.5 · shaping'},
   {id:'chk', col:0,row:6,kind:'data',label:'checkpoint', op:'planned · potential'},
   {id:'comp',col:1,row:0,kind:'loss',label:'compose_reward', op:'weighted sum', h:7},
   {id:'rew', col:2,row:0,kind:'data',label:'team reward rₜ', h:7}
 ], edges:[
   {from:'cov',to:'comp'},{from:'conn',to:'comp'},{from:'coll',to:'comp'},
   {from:'anti',to:'comp'},{from:'info',to:'comp'},{from:'cong',to:'comp'},
   {from:'chk',to:'comp',kind:'soft'},{from:'comp',to:'rew'}
 ]},
 detail:`
 <p>The simulator runs <b>reward-agnostic</b> — it emits raw per-term magnitudes and the experiment defines the
 mission by weighting them in <code>compose_reward</code>. Everything is scored at the <b>team</b> level
 (Dec-POMDP shared reward). Default = <code>1·coverage + 2·connectivity − 4·collision</code>.</p>
 <table class="spec">
  <tr><th>term</th><th>× default</th><th>what / why</th><th>class</th></tr>
  <tr><td>coverage</td><td>1.0</td><td>fresh team-covered cells — the deliverable</td><td>objective</td></tr>
  <tr><td>connectivity</td><td>2.0</td><td>fraction of others reachable — weighted 2× as the binding, fragile constraint</td><td>objective</td></tr>
  <tr><td>collision</td><td>−4.0</td><td>co-located agents — a large <em>fixed</em> anti-crowding guardrail</td><td>guardrail</td></tr>
  <tr><td>anti-overlap</td><td>1.0</td><td>− simultaneous double-coverage → pushes division of labour</td><td>shaping</td></tr>
  <tr><td>info-gain</td><td>0.1</td><td>+ uncovered cells in range → exploration bonus (a null lever in our ablations)</td><td>shaping</td></tr>
  <tr><td>congestion</td><td>0.5</td><td>− neighbours picking the same skill → anti-collapse price</td><td>shaping</td></tr>
  <tr><td>soft-λ</td><td>dial</td><td>−penalty·λ₂-shortfall (the cov↔conn dial, when mechanism=soft_lambda)</td><td>constraint</td></tr>
 </table>
 <h4>How we got here</h4>
 <p>• <b>Reward-agnostic engine.</b> Keeping the reward out of the physics lets one simulator serve
 Sense/Organize/Act missions by re-weighting — the experiment, not the world, picks the objective.<br>
 • <b>Team-level, not per-agent.</b> coverage and connectivity are team quantities; per-agent credit is handled by
 <em>shaping</em> (anti-overlap) and, for measurement, exact difference rewards — never by splitting the reward.<br>
 • <b>Objective vs shaping.</b> coverage + connectivity + collision <em>are</em> the mission; the rest are denser
 signals to guide learning, kept small so they don't move the optimum (and pushed toward potential-based forms).<br>
 • <b>The 2:1 connectivity prior.</b> Connectivity is the constraint and the easy-to-lose property, so it's weighted
 higher — but it's a <em>dial</em>: the frontier sweep traces the whole trade-off (raise w-coverage → shed connectivity).<br>
 • <b>Collision −4, fixed.</b> A behavioural guardrail big enough to dominate locally; a hard rule, not a mission knob, so it isn't exposed on the CLI.<br>
 • <b>Honest ablations.</b> anti-overlap helped (division of labour — took CTDE past 90% once); info-gain and
 SLAM-coverage were null levers — kept available, weighted low.</p>
 <div class="note"><b>Incoming — the checkpoint "prize."</b> The arena reward for clearing a gated corridor will land
 as <b>potential-based shaping</b> <code>F = γΦ(s′) − Φ(s)</code> (Ng et al. 1999): <b>policy-invariant</b>, so it
 speeds discovery of the chain-through-corridor behaviour <em>without</em> changing the optimum or being mistaken for
 emergence (the reviewer-kill trap). Φ = fraction of the checkpoint cleared. The alternative is a true
 <em>mission</em> term (delivered-coverage) — a knowing choice, not a hand-tuned dense bonus.</div>
 <p class="cite">env_utils.py:106-157 (compose_reward) · config.py Reward · checkpoint = next build</p>`},

/* ============================ TRAINING ============================ */
{id:'ppo', name:'PPO training loop', group:'Training', badge:'learned',
 tagline:'Rollout → GAE → clipped policy + value + aux − entropy → AdamW → back to rollout.',
 graph:{ nodes:[
   {id:'roll',col:1,row:0,kind:'data',label:'rollout buffer', op:'obs,act,logp,rew,val'},
   {id:'gae', col:1,row:1,kind:'data',label:'GAE', sub:'γ.99 λ.95 → Â'},
   {id:'tot', col:1,row:2,kind:'loss',label:'total loss', op:'PPO + ½·value + aux − ent'},
   {id:'adam',col:1,row:3,kind:'ctrl',label:'AdamW', op:'lr 3e-4 · clip‖g‖.5'},
   {id:'th',  col:1,row:4,kind:'head',label:'θ actor+critic'}
 ], edges:[
   {from:'roll',to:'gae'},{from:'gae',to:'tot',label:'Â'},
   {from:'tot',to:'adam',kind:'grad'},{from:'adam',to:'th',kind:'grad'},
   {from:'th',to:'roll',kind:'feedback',route:'margin',label:'act'}
 ]},
 detail:`
 <p>Total loss = <code>PPO(goal) + 0.5·value + 0.1·aux(λ̂₂) + degree_reg·Var(mean-degree) − 0.01·(goal+role+skill entropy)</code>.
 8 parallel rollouts/iter, horizon 100, 4 epochs × 4 minibatches, AdamW (decoupled weight decay 1e-4).</p>
 <table class="spec">
  <tr><th>hp</th><th>default</th><th>hp</th><th>default</th></tr>
  <tr><td>lr</td><td>3e-4</td><td>clip</td><td>0.2</td></tr>
  <tr><td>γ / gae-λ</td><td>0.99 / 0.95</td><td>entropy</td><td>0.01</td></tr>
  <tr><td>epochs</td><td>4</td><td>minibatches</td><td>4</td></tr>
  <tr><td>rollouts/iter</td><td>8</td><td>horizon</td><td>100</td></tr>
 </table>
 <p class="cite">ppo.py:16-708 · train_ctde.py:46-237</p>`},

{id:'loss', name:'Loss & regularization — why each term', group:'Training', badge:'learned',
 tagline:'Standard PPO machinery + three deliberate additions — and the connectivity constraint lives in the reward, not here.',
 graph:{ nodes:[
   {id:'ppo',col:0,row:0,kind:'loss',label:'PPO clip · goal', op:'policy objective'},
   {id:'val',col:0,row:1,kind:'loss',label:'value MSE', op:'× 0.5'},
   {id:'aux',col:0,row:2,kind:'loss',label:'aux λ̂₂', op:'× 0.1'},
   {id:'ent',col:0,row:3,kind:'loss',label:'− entropy', op:'× 0.01'},
   {id:'deg',col:0,row:4,kind:'loss',label:'degree-reg', op:'× 1e-3'},
   {id:'tot',col:1,row:0,kind:'head',label:'total loss L', h:5},
   {id:'opt',col:2,row:2,kind:'ctrl',label:'AdamW', op:'lr 3e-4 · wd 1e-4 · ‖g‖ .5'}
 ], edges:[
   {from:'ppo',to:'tot'},{from:'val',to:'tot'},{from:'aux',to:'tot'},{from:'ent',to:'tot'},{from:'deg',to:'tot'},
   {from:'tot',to:'opt',kind:'grad'}
 ]},
 detail:`
 <p><b>The objective</b> — minimised with AdamW, advantages from GAE (γ=0.99, λ=0.95):</p>
 <p style="background:#f4f7fb;border:1px solid var(--line);border-radius:7px;padding:9px 12px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.6">
 L = L<sup>PPO</sup>(goal) &nbsp;+&nbsp; 0.5·L<sub>value</sub> &nbsp;+&nbsp; 0.1·L<sub>aux</sub>(λ̂₂) &nbsp;+&nbsp; 1e-3·Var(mean-degree) &nbsp;−&nbsp; 0.01·(H<sub>goal</sub>+H<sub>role</sub>+H<sub>skill</sub>)</p>
 <table class="spec">
  <tr><th>term</th><th>what it computes</th><th>why it's here</th><th></th></tr>
  <tr><td>PPO clip (goal)</td><td>clipped importance-sampling surrogate on the goal policy</td><td>stable on-policy updates — clip is a cheap trust region</td><td>PPO</td></tr>
  <tr><td>value MSE ×0.5</td><td>critic V vs GAE returns</td><td>low-variance baseline; a <em>central</em> critic = CTDE/MAPPO</td><td>MAPPO</td></tr>
  <tr><td>aux λ̂₂ ×0.1</td><td>predict local Fiedler λ₂ (supervised vs oracle)</td><td>auxiliary task → forces belief z to encode graph structure</td><td>ours</td></tr>
  <tr><td>entropy ×0.01</td><td>−H over goal + role + skill</td><td>keep exploration alive at <em>every</em> level of the hierarchy</td><td>ours</td></tr>
  <tr><td>degree-reg ×1e-3</td><td>batch variance of the team's mean comm-degree</td><td>reward <em>consistent</em> connectivity, not just average</td><td>ours</td></tr>
 </table>
 <h4>How we got here</h4>
 <p>• <b>PPO, not vanilla PG or off-policy.</b> The clip gives a trust region for free (stable updates), and on-policy
 fits the vmap'd parallel JAX rollouts and the single cooperative return.<br>
 • <b>Central critic (CTDE/MAPPO).</b> The value term trains a god-view critic so the GAE baseline sees the whole
 team — cutting the variance that independent per-agent critics suffer under multi-agent non-stationarity. An
 IPPO/decentral-critic fallback is kept.<br>
 • <b>The connectivity constraint is NOT a loss term.</b> Following constrained-MARL (RCPO / Lagrangian-PPO), the
 constraint enters through the <em>reward</em> via a running dual multiplier (see the connectivity dual loop) — so L
 stays standard PPO and the pressure adapts online. Loss and constraint stay separable.<br>
 • <b>Auxiliary λ̂₂ head.</b> A UNREAL-style auxiliary task: predicting the Fiedler value shapes the backbone toward
 a connectivity-aware representation the policy needs — and gives the contribution/diagnostic estimator for free.
 Weighted small (0.1) so it shapes, not dominates.<br>
 • <b>Entropy summed over the hierarchy.</b> goal, role and skill are three categorical heads that can each collapse;
 one entropy term per level keeps all of them exploring, not just the goal.<br>
 • <b>degree-reg</b> is the one bespoke regulariser: penalise the <em>swing</em> in team connectivity (its variance),
 while the <em>mean</em> is handled by the reward/dual. The mission needs reliable graphs, so we regularise reliability.</p>
 <h4>Regularization at a glance</h4>
 <table class="spec">
  <tr><th>knob</th><th>value</th><th>job</th></tr>
  <tr><td>entropy coef</td><td>0.01</td><td>exploration / anti-collapse (all 3 heads)</td></tr>
  <tr><td>weight decay (AdamW)</td><td>1e-4</td><td>generalisation (decoupled L2)</td></tr>
  <tr><td>grad-norm clip</td><td>0.5</td><td>training stability</td></tr>
  <tr><td>degree-reg</td><td>1e-3</td><td>connectivity consistency</td></tr>
  <tr><td>dropout</td><td>0 (off)</td><td>belief is already noisy (partial-obs + gossip)</td></tr>
  <tr><td>GAE λ</td><td>0.95</td><td>advantage bias ↔ variance</td></tr>
 </table>
 <div class="note"><b>What we deliberately left out:</b> no COMA/counterfactual credit in the loss (too noisy for
 sparse coverage — see the Critic note) · no connectivity penalty <em>directly</em> in L (it's in the reward via the
 dual) · no KL-penalty PPO variant (the clip suffices).</div>
 <p class="cite">ppo.py:16-17, 703-708 · config.py Loss / Regularization / Trainer</p>`},

{id:'es', name:'ES coexistence (MERL)', group:'Training', badge:'learned',
 tagline:'Evolution perturbs the small selector by team fitness while PPO gradient-trains the executor.',
 graph:{ nodes:[
   {id:'th',  col:1,row:0,kind:'head',label:'selector θ', op:'small'},
   {id:'pert',col:1,row:1,kind:'data',label:'θ ± σε', op:'pop 16 antithetic'},
   {id:'roll',col:1,row:2,kind:'env', label:'team rollouts'},
   {id:'fit', col:1,row:3,kind:'data',label:'team return', op:'rank-shaped f̃'},
   {id:'upd', col:1,row:4,kind:'ctrl',label:'θ ← θ + lr/Pσ·Σf̃ε'},
   {id:'exec',col:0,row:2,kind:'head',label:'PPO executor', op:'gradient · disjoint leaves'}
 ], edges:[
   {from:'th',to:'pert'},{from:'pert',to:'roll'},{from:'roll',to:'fit'},{from:'fit',to:'upd'},
   {from:'upd',to:'th',kind:'feedback',route:'margin',label:'ES step'},
   {from:'exec',to:'roll',label:'shared rollout'}
 ]},
 detail:`
 <p>The two learners touch <b>disjoint parameter leaves</b> and share only the team return — timescale-separated
 (ES per generation, PPO per step). Defaults: <code>pop 16 · σ 0.05 · lr 0.05 · kind nes · elite_frac 0.25</code>.
 NES update: <code>θ ← θ + (lr/Pσ)·Σ f̃ᵢ·εᵢ</code> with rank-shaped fitness.</p>
 <p class="cite">es.py:120-334 (merl_coexist 272-334)</p>`}

];
