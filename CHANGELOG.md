# CHANGELOG

<!-- version list -->

## v2.5.0 (2026-07-01)

### Bug Fixes

- Improve the pop up window for the plan graph
  ([`4b407d6`](https://github.com/AI4MS/MatCreator/commit/4b407d6406352d7ef618dbb6d340c20bfc5101f4))

- Local user has full access to session logs
  ([`9ecc047`](https://github.com/AI4MS/MatCreator/commit/9ecc047ad8a908adad32f35d2bf3034df655ed28))

### Features

- Add llm cards
  ([`1e97a26`](https://github.com/AI4MS/MatCreator/commit/1e97a26299c8d7075b0d99f9efa54258d2473768))


## v2.4.0 (2026-06-30)

### Bug Fixes

- Auto-wrap executor title on card expand, keep triangle aligned to first line
  ([`be8e8fa`](https://github.com/AI4MS/MatCreator/commit/be8e8faf00e77ed779687fccbb17e50f6690320c))

- Improve front end UI
  ([`85cc466`](https://github.com/AI4MS/MatCreator/commit/85cc466f99b2aba6db8dae8766194c42c9487df2))

- Issue with the ordering of sub-executor
  ([`7556ef5`](https://github.com/AI4MS/MatCreator/commit/7556ef5636ed00ef3eeffe2a003e107764274f07))

### Features

- Add crash recovery support
  ([`f1671d5`](https://github.com/AI4MS/MatCreator/commit/f1671d51bb92351a5d73719b10801da9f4b70f1b))

- Image lightbox with zoom, pan, and drag for file viewer and chat images
  ([`60cc250`](https://github.com/AI4MS/MatCreator/commit/60cc250b6f334dc59209cf904a30b5416f6ec52f))

- Split plan graph into navigable subgraphs; fallback to manual grid when no edges
  ([`039a8cc`](https://github.com/AI4MS/MatCreator/commit/039a8ccf553363ee29976dcb1910e63b7a7f503e))

- Unify the session log
  ([`cf5a531`](https://github.com/AI4MS/MatCreator/commit/cf5a531384a9c19caa87643a31765a02810a3435))


## v2.3.0 (2026-06-29)

### Bug Fixes

- Frontend CLI now support login bash
  ([`29d3756`](https://github.com/AI4MS/MatCreator/commit/29d3756f201768cc00a7905f7bf211cc3688755c))

- Refactor the frontend
  ([`7e89724`](https://github.com/AI4MS/MatCreator/commit/7e897246eec58d0992e343856d5842bd1159e105))

- Too many details in search_skills
  ([`00ee005`](https://github.com/AI4MS/MatCreator/commit/00ee0052408cd97a5692fe6e1c4228a23f004cb8))

- Update docker file
  ([`fedcd4e`](https://github.com/AI4MS/MatCreator/commit/fedcd4e768076de4b16b99bf828ed5749b2420c4))

- Update dockerfile
  ([`aab59ba`](https://github.com/AI4MS/MatCreator/commit/aab59ba242dc0d4e6f94a622dca85661f8e3bc05))

### Documentation

- **cli**: Expand help text and add metavar to CLI arguments
  ([`1b61f50`](https://github.com/AI4MS/MatCreator/commit/1b61f5031048d0e8cc1dc756ba22356f2460fa04))

- **ports**: Document port configuration in deployment guides
  ([`cd62c9e`](https://github.com/AI4MS/MatCreator/commit/cd62c9e55db555d0d8c91ba717ff433d84cb67e6))

### Features

- Add rounded favicon for browser tab
  ([`8779245`](https://github.com/AI4MS/MatCreator/commit/8779245003d004178fe3e22a39ccfdda55253d9f))

- **deploy**: Make Docker Compose and nginx ports configurable
  ([`d6f9333`](https://github.com/AI4MS/MatCreator/commit/d6f933365ddf14ea27c0f6156ac010e1d2143158))

- **ports**: Add centralized port configuration module
  ([`0060073`](https://github.com/AI4MS/MatCreator/commit/0060073cb2098dee18230038bc647cf06dfec56a))

- **ports**: Add configurable host resolution for core services
  ([`0b4f885`](https://github.com/AI4MS/MatCreator/commit/0b4f885eae715744b64b563bff3ecba86582d0aa))

- **ports**: Wire port config into application code
  ([`81b6f23`](https://github.com/AI4MS/MatCreator/commit/81b6f236d0badaf60a24a74a64fe5f2cba8e0be6))

### Testing

- **ports**: Add comprehensive port configuration tests
  ([`206373d`](https://github.com/AI4MS/MatCreator/commit/206373d527dec3b4e37deb66689c98b18ab7eb28))


## v2.2.0 (2026-06-25)

### Bug Fixes

- Interleave step cards with messages via unified timeline in loadSession
  ([`38ca1c7`](https://github.com/nlz25/PFD_Agent/commit/38ca1c7c61c5dd8eb33e418b42adc1ca48452a21))

- Remove the tester agent route
  ([`45af7d6`](https://github.com/nlz25/PFD_Agent/commit/45af7d64a50d57926be5d59599efe10c1ee226d5))

- Restore unified timeline interleaving, remove session summary refs
  ([`60713fb`](https://github.com/nlz25/PFD_Agent/commit/60713fb8dea40ae390a84d6b8f74cc62d1d85263))

- SSE stream parsing, ASCII art rendering, table styles, and timeline ordering
  ([`05534ee`](https://github.com/nlz25/PFD_Agent/commit/05534eebe3ea57b7a7ffbb1210497d8fedc4e84a))

### Features

- Json-block escape handling, wrap markers, and IN/OUT badges
  ([`66d6025`](https://github.com/nlz25/PFD_Agent/commit/66d60259b9ad4062d8b580a3d9e88df913b1b2b0))

- **frontend**: Improve plan graph layout, collapsible sections, and node ordering
  ([`ad3acae`](https://github.com/nlz25/PFD_Agent/commit/ad3acae6d6e39d5eb827516d9d8e7ad876fa48ae))


## v2.1.0 (2026-06-22)

### Bug Fixes

- Add nginx server config and server mode doc
  ([`028ac3e`](https://github.com/nlz25/PFD_Agent/commit/028ac3e6256d86a8f0cb8d998457c9392149345f))

- Issues with log directory and custom workdir
  ([`8d10980`](https://github.com/nlz25/PFD_Agent/commit/8d109802b3a60b4ed2ec6221ee1d6be192f446bb))

- Move sub-agent process to main conversation
  ([`aba1514`](https://github.com/nlz25/PFD_Agent/commit/aba151484dd869ebf7271397c44bea335fb5fb6f))

- Restore dpa4 skill and fix virial data handling
  ([`c35e5c7`](https://github.com/nlz25/PFD_Agent/commit/c35e5c7d13e847c4e2e1736785ddaf50ba812eea))

### Features

- Add workspace CLI interface
  ([`a2faf21`](https://github.com/nlz25/PFD_Agent/commit/a2faf2138943f2558fb45e176ceef0721c7dff79))

- Implement multi-user mode
  ([`d155022`](https://github.com/nlz25/PFD_Agent/commit/d155022e72a6d452720d7d4aac0e725f01338d9a))


## v2.0.0 (2026-06-18)

### Bug Fixes

- Correct pfactor parameter for ase NPT modes
  ([`13c1e33`](https://github.com/nlz25/PFD_Agent/commit/13c1e33e242a37b1e58f81e08d5e91e490370d6f))

- Improve user management
  ([`d8261c5`](https://github.com/nlz25/PFD_Agent/commit/d8261c5412d6c3e1afc3e1db63848c3b0e453e74))

- Issues about kdg progressive retrieval
  ([`e50be7d`](https://github.com/nlz25/PFD_Agent/commit/e50be7dc8d0a14dbfe7f2b8dcaf3b17c92fdf005))

- Issues with step executor and graph logger
  ([`d11aa02`](https://github.com/nlz25/PFD_Agent/commit/d11aa020dbe8565c6ccd1a4c922b78155beaea8e))

- Issues with the know_do_graph initialization
  ([`c213ccf`](https://github.com/nlz25/PFD_Agent/commit/c213ccf0c94c958cf522124903f0dcd8a509cadd))

- Kdg frontend use wrong path
  ([`7f2e0fc`](https://github.com/nlz25/PFD_Agent/commit/7f2e0fc8c748d0564afc1a8fff70c900bd0339c3))

- Legacy path issues for kdg
  ([`548d2f5`](https://github.com/nlz25/PFD_Agent/commit/548d2f54d7945c4822be7c9bb5f257a35362cb7a))

- Remove old ase-deepmd model and relax results
  ([`28e4b9e`](https://github.com/nlz25/PFD_Agent/commit/28e4b9e142a1d8a4c60e4aa61dfbf4aeab042e27))

- Resolve merge conflicts between devel and refactor branches
  ([`8b20581`](https://github.com/nlz25/PFD_Agent/commit/8b205814bb18920d3d3dc62622475302667ce351))

- Review agent bugs
  ([`bdc37ab`](https://github.com/nlz25/PFD_Agent/commit/bdc37ab7c33dd3988d3155ae324040870710acbf))

- Update front end
  ([`3bbc01f`](https://github.com/nlz25/PFD_Agent/commit/3bbc01f3cd6a7ced0a8256760f7938674232d6b5))

- Update history tools
  ([`b06fc40`](https://github.com/nlz25/PFD_Agent/commit/b06fc4014166522ced88a9f54016fc463f3cc538))

- Update import paths in test files
  ([`cb50b08`](https://github.com/nlz25/PFD_Agent/commit/cb50b08bbda0b35f97dcb044bbe77dcbbbbc3ea8))

- Use new kdg version for optional download
  ([`0fa0ad0`](https://github.com/nlz25/PFD_Agent/commit/0fa0ad0eadf788c0e6466aadd42d73433068f6d5))

- **dpa4**: Fix YAML frontmatter — move MANDATORY block out of frontmatter
  ([`53b0276`](https://github.com/nlz25/PFD_Agent/commit/53b02765e5237565d549188a9788a62860501e22))

- **dpa4**: Improve workflow clarity and enforce skill authority
  ([`da9fc03`](https://github.com/nlz25/PFD_Agent/commit/da9fc03de88122e36f09f0aa4b250e86aa15c4d6))

### Features

- Cli for graph frontend.
  ([`5a31b58`](https://github.com/nlz25/PFD_Agent/commit/5a31b588620b96bf0e9a4d29674eac37011d1350))

- **dpa4**: Add DPA4 (SeZM/neo) finetuning skill
  ([`cf32ae6`](https://github.com/nlz25/PFD_Agent/commit/cf32ae60cebef1d4b7a80d1e4bc14350248e206c))

### Refactoring

- Refactor codebase structure
  ([`bb983c4`](https://github.com/nlz25/PFD_Agent/commit/bb983c406f10d3f0f28e4e9b2ff1fd4682adeb33))

- **dpa4**: Rewrite workflow — DFT labeling & benchmarks, remove zero-shot
  ([`b15399a`](https://github.com/nlz25/PFD_Agent/commit/b15399a4f24fff0ba13ccae2645b6e9aeee384fc))


## v1.12.0 (2026-06-14)

### Bug Fixes

- Remove plan tools in flash mode
  ([`7552ce3`](https://github.com/nlz25/PFD_Agent/commit/7552ce3f3b7e8544f13213561502d19bfbe3ae2e))

- Update agent instructions
  ([`f7c1527`](https://github.com/nlz25/PFD_Agent/commit/f7c152734b9fc87c48c8f697de042b5536446722))

### Features

- Adding know-do graph
  ([`dabfd15`](https://github.com/nlz25/PFD_Agent/commit/dabfd15c8edf0dfb3fe60a39511741d6526094c1))

- Auto review graph
  ([`d34a7fa`](https://github.com/nlz25/PFD_Agent/commit/d34a7fa598f3edc39c29e9cb44d3c2d58c8d0866))

- Graph review
  ([`62c511f`](https://github.com/nlz25/PFD_Agent/commit/62c511f13657df6c8f381737bf0663a20a97cf21))


## v1.11.0 (2026-06-10)

### Bug Fixes

- **lammps**: Address review comments — move submission to references, use bohrium skill
  ([`1b07f7a`](https://github.com/nlz25/PFD_Agent/commit/1b07f7a027507560a46f8dd6bd9bfe866ec6b4f2))

### Features

- **session**: Add session delete feature
  ([`9f762ac`](https://github.com/nlz25/PFD_Agent/commit/9f762acff25df94c5c08ce811801cc541e81b6d9))


## v1.10.0 (2026-06-09)

### Bug Fixes

- Issues with graph glitch and skill addition
  ([`d5fbc2a`](https://github.com/nlz25/PFD_Agent/commit/d5fbc2a34be0a0b2c6baf363bf3e00dec82cb2cb))

- Issues with user login
  ([`d484a7d`](https://github.com/nlz25/PFD_Agent/commit/d484a7d17f923702f552a3dc263145a8dc317ce6))

- Minor issues
  ([`1421b1d`](https://github.com/nlz25/PFD_Agent/commit/1421b1d6160a2fdbd77ab2548009531ad4379b8d))

- Update instructions for flash mode
  ([`0e9f2ff`](https://github.com/nlz25/PFD_Agent/commit/0e9f2ff96fc21a1d56db993eea6f65d51e27452d))

- **embedding**: Add drop_params=True to suppress UnsupportedParamsError for minimax
  ([`09a5000`](https://github.com/nlz25/PFD_Agent/commit/09a50006d6e7005628b9e41d690d44828b398547))

### Features

- Add new materials modelling skill 'matcraft-kit'
  ([`a8b1c1c`](https://github.com/nlz25/PFD_Agent/commit/a8b1c1c907a94be0c5f205be495b6f8e35c89acf))

- **lammps**: Add LAMMPS skill for DeepMD-based MD simulations
  ([`3a3d774`](https://github.com/nlz25/PFD_Agent/commit/3a3d77495e6789b751796cfcd3be1bf57f341a72))


## v1.9.0 (2026-06-08)

### Bug Fixes

- Add new vasp and bohr skill
  ([`f50bca8`](https://github.com/nlz25/PFD_Agent/commit/f50bca89c648123a4196e3ed6edf4058dd8a5156))

- Disabled skills now behave properly
  ([`9947bf5`](https://github.com/nlz25/PFD_Agent/commit/9947bf54cf1923bc621cc896c05473b496407f0a))

- Improve CLI mode
  ([`7aadad0`](https://github.com/nlz25/PFD_Agent/commit/7aadad02cae65650ee72a3fe0b2420a98a78a15b))

- Minor issue
  ([`01e9da1`](https://github.com/nlz25/PFD_Agent/commit/01e9da14aeb5133af9c435c1c4b324c84d8aa866))

- Update legacy VASP skill
  ([`3ff1047`](https://github.com/nlz25/PFD_Agent/commit/3ff1047f901d9ace4a8c88fcaeaf986a9afd511e))

- Update vasp-pymatgen skill
  ([`885d4bc`](https://github.com/nlz25/PFD_Agent/commit/885d4bc7acea626a5654569468be3f6132e514c9))

### Features

- Add custom skill management
  ([`6c33ad0`](https://github.com/nlz25/PFD_Agent/commit/6c33ad0e26a30e5b37eae837aefe699c57672537))


## v1.8.0 (2026-06-04)

### Bug Fixes

- Update the frontend layout
  ([`8493bb1`](https://github.com/nlz25/PFD_Agent/commit/8493bb190c950e49b986b13f83240aee7adeab9c))

### Features

- Add flash mode and improved frontend
  ([`fd30da8`](https://github.com/nlz25/PFD_Agent/commit/fd30da8876146ac35f61ad2a2ba4ea10381a03f8))


## v1.7.4 (2026-05-28)

### Bug Fixes

- Update skills and description
  ([`2c62d00`](https://github.com/nlz25/PFD_Agent/commit/2c62d00a42e2082bb5f05d82ce781d450e2c901e))


## v1.7.3 (2026-05-27)

### Bug Fixes

- Add get_related_skills to execution agent
  ([`b82e445`](https://github.com/nlz25/PFD_Agent/commit/b82e445bdc63d87d948a71fd7b206383850798e1))


## v1.7.2 (2026-05-27)

### Bug Fixes

- Issues with refresh_skill tool
  ([`fd58a24`](https://github.com/nlz25/PFD_Agent/commit/fd58a243f00b266391d0f3ccf341cda8b08803db))


## v1.7.1 (2026-05-27)

### Bug Fixes

- Issues the project root solving logic
  ([`1e3d3f1`](https://github.com/nlz25/PFD_Agent/commit/1e3d3f1f9c0871b1f9f3364b4528353f9107aaf0))

- Update src/matcreator/scripts/start_agent.py
  ([`4cf235b`](https://github.com/nlz25/PFD_Agent/commit/4cf235b22ddae94da085daecb241e634cc066d67))


## v1.7.0 (2026-05-26)

### Bug Fixes

- Add script for graph visualization
  ([`a1ad34f`](https://github.com/nlz25/PFD_Agent/commit/a1ad34f97f82c54817e883f358957c83d295f256))

### Features

- Add skill graph
  ([`db44113`](https://github.com/nlz25/PFD_Agent/commit/db441130f39829e313c8173bcaa29cec22f3001b))

- Graph planning
  ([`48fa5eb`](https://github.com/nlz25/PFD_Agent/commit/48fa5ebda5cb2e647f34bc160e0c8ad79ba76b23))


## v1.6.1 (2026-05-23)

### Bug Fixes

- Add MP skill
  ([`c025e4d`](https://github.com/nlz25/PFD_Agent/commit/c025e4d8dcfbc5001f39b34f53e4b10292d4f32b))

- Decode output in run_bash to handle byte strings correctly
  ([`eb194eb`](https://github.com/nlz25/PFD_Agent/commit/eb194ebbba12c37e9a634ca891601af69b04d0ef))

- Issues with sub-steps within the executor
  ([`abcd5f6`](https://github.com/nlz25/PFD_Agent/commit/abcd5f6d0b6e188e8273a3299fa9e46c0ec42743))

- Issues with the validate_plan tool
  ([`4da2fa8`](https://github.com/nlz25/PFD_Agent/commit/4da2fa849f2558e6330b11511d0fea7da3e03383))

- Update run_skill_script tool
  ([`665a87d`](https://github.com/nlz25/PFD_Agent/commit/665a87d680bca28255edc275e51ecfcfa6f75f1a))

- Update step executor guidelines
  ([`b57f080`](https://github.com/nlz25/PFD_Agent/commit/b57f0801bcc2c5faa0910e9fc244533ec227d8e1))


## v1.6.0 (2026-05-20)

### Bug Fixes

- Improve the benchmark mode
  ([`cb31245`](https://github.com/nlz25/PFD_Agent/commit/cb312454141c435d15ac3092741404438d673e9e))

### Features

- Sub-step decomposition for execution agent
  ([`a542d94`](https://github.com/nlz25/PFD_Agent/commit/a542d942f9d9226f54a706c7c829f287a33ea48b))


## v1.5.2 (2026-05-18)

### Bug Fixes

- Add history review tools and cancelllation
  ([`7842fc7`](https://github.com/nlz25/PFD_Agent/commit/7842fc788197570d3466d3c0402d8f03b924afe9))

- Coarse-grained planning
  ([`6428033`](https://github.com/nlz25/PFD_Agent/commit/642803348aee248a2243b63333078ddc1d8cd58e))

- Minor issues with planning
  ([`7f2fa22`](https://github.com/nlz25/PFD_Agent/commit/7f2fa2275898aea27cd8e6f29416caeac6badf3f))

- Update dpdisp skill
  ([`3ba73a4`](https://github.com/nlz25/PFD_Agent/commit/3ba73a47eb79660970b448bd52befde8af729bf4))

- Update README
  ([`c1d1aad`](https://github.com/nlz25/PFD_Agent/commit/c1d1aad27e86b50c79c03f0b1a169d5649beb5b4))


## v1.5.1 (2026-05-18)

### Bug Fixes

- Update README
  ([`1f04ed4`](https://github.com/nlz25/PFD_Agent/commit/1f04ed47ea2dd7d3fda82d2bdcaeebfee222503f))


## v1.5.0 (2026-05-15)

### Bug Fixes

- Add admin user
  ([`8707aa0`](https://github.com/nlz25/PFD_Agent/commit/8707aa038d08a9babc7e473957568d837751dd8b))

- Deduplicate LLM response
  ([`12e35fa`](https://github.com/nlz25/PFD_Agent/commit/12e35fa4ab5b14f782e6fac74ad8ac53af9177b3))

- Improved frontend UI
  ([`e2a04b8`](https://github.com/nlz25/PFD_Agent/commit/e2a04b82ea1d815df7312a487b5451337fb23d1b))

- Refine agent_graph, function flow ,picture display
  ([`392df63`](https://github.com/nlz25/PFD_Agent/commit/392df63b0fb9f3b963aa52e6e515760a947cc15d))

### Features

- Graph based memory system
  ([`5208601`](https://github.com/nlz25/PFD_Agent/commit/5208601e73bd50e5bede55c2eb7cb363deb8f192))


## v1.4.0 (2026-05-12)

### Bug Fixes

- Update frontend
  ([`7e5593a`](https://github.com/nlz25/PFD_Agent/commit/7e5593a7b74cc159530f5d77b8cc8ae8177aec51))

### Features

- Restructure skill system
  ([`764160c`](https://github.com/nlz25/PFD_Agent/commit/764160c30f1e1be4f129966fa8a935ebfd026a3a))


## v1.3.0 (2026-05-11)

### Bug Fixes

- Add quick start script for matcreator
  ([`4f84de5`](https://github.com/nlz25/PFD_Agent/commit/4f84de53d92990f2677de1efce2a00701202bd6d))

- Add tavily skill for web-search
  ([`69c5101`](https://github.com/nlz25/PFD_Agent/commit/69c51018d4e3f36773b6eade7f2c67b05753174c))

- Issues with repeated plan id
  ([`5ac7ab7`](https://github.com/nlz25/PFD_Agent/commit/5ac7ab7936ce80c3e4ca327ca9384161c7779ebb))

- New frontend
  ([`7e4be39`](https://github.com/nlz25/PFD_Agent/commit/7e4be39b1c95fc9fc542a6cdb1068e944d6c68b6))

- Refine mattergen and subagent
  ([`1c24735`](https://github.com/nlz25/PFD_Agent/commit/1c24735a3f31bc01a936aae1453456c74e66af63))

- UI unsafe path handling
  ([`95ec643`](https://github.com/nlz25/PFD_Agent/commit/95ec6432452d8c0085ada6b1864dbf11ca3e24e1))

- Update frontend
  ([`0e7bea7`](https://github.com/nlz25/PFD_Agent/commit/0e7bea7439ee8afa186660f2cac4aa6b6a738ecb))

- Update UI
  ([`e395a01`](https://github.com/nlz25/PFD_Agent/commit/e395a01fe2b4495a8fc44662a08d2fd424c28c25))

### Features

- Add new front end
  ([`ce93220`](https://github.com/nlz25/PFD_Agent/commit/ce932207203cc1955d24a1d3317f6f73e2fd3630))

- Parallel sub-agent organization
  ([`54b9b3d`](https://github.com/nlz25/PFD_Agent/commit/54b9b3d3cd85077d83906e142b528d099bd9c32c))


## v1.2.4 (2026-05-06)

### Bug Fixes

- Isolate execution context
  ([`0017d2f`](https://github.com/nlz25/PFD_Agent/commit/0017d2fc8ca295bd78614124fd46802e2b0c8575))


## v1.2.3 (2026-04-23)

### Bug Fixes

- Add resume tool for interrupted execution
  ([`bacb2c0`](https://github.com/nlz25/PFD_Agent/commit/bacb2c09b188af07f5ab8d0951b77026647ab3b9))

- Rescope summarize and intent tools
  ([`b8d1070`](https://github.com/nlz25/PFD_Agent/commit/b8d107027a44349d93eed9eca6634f6462fc08bd))

- Update UI for better streaming
  ([`a60a2ad`](https://github.com/nlz25/PFD_Agent/commit/a60a2ad8d9869d5a278fe3296644f1114b38f21e))


## v1.2.2 (2026-04-21)

### Bug Fixes

- Add custom session for CLI
  ([`bdb2e07`](https://github.com/nlz25/PFD_Agent/commit/bdb2e0703f036b708b4794bc0113d6df03744a54))

- Update app UI
  ([`3e2c85d`](https://github.com/nlz25/PFD_Agent/commit/3e2c85d096932baf8dbc4a42398e205eaef32c30))

- Update skill
  ([`5d43e8a`](https://github.com/nlz25/PFD_Agent/commit/5d43e8a666b85457520c0634e2b381f4d7c46e57))

- Version of a2a-sdk to 0.3.25
  ([`bb65106`](https://github.com/nlz25/PFD_Agent/commit/bb65106016ef9ca1bbda1e498ba780619e69b5e7))


## v1.2.1 (2026-04-20)

### Bug Fixes

- Add mattersim skill
  ([`7015bdd`](https://github.com/nlz25/PFD_Agent/commit/7015bdd58c873068b39977de866f4c1211e1c6fc))

- Update CLI
  ([`7e7ce8b`](https://github.com/nlz25/PFD_Agent/commit/7e7ce8b8d2ade2b78e856e9ddd9a5532775db286))


## v1.2.0 (2026-04-19)

### Bug Fixes

- Update skill structure
  ([`9cd18da`](https://github.com/nlz25/PFD_Agent/commit/9cd18da30901c600a472cd7bb786942303c4ba59))

### Features

- Add non-interactive CLI mode
  ([`e5de004`](https://github.com/nlz25/PFD_Agent/commit/e5de004973b4ca7393b34897a1f4370bdf5156c9))


## v1.1.1 (2026-04-15)

### Bug Fixes

- Restore the bash tools for thinking agent
  ([`aa485a9`](https://github.com/nlz25/PFD_Agent/commit/aa485a97a196b06da8948f5c384866870fe7c765))


## v1.1.0 (2026-04-15)

### Bug Fixes

- Add unittest
  ([`e0cff24`](https://github.com/nlz25/PFD_Agent/commit/e0cff24b5250e1e4bd72bc63913c18578d16ac2f))

- Change skills location
  ([`01d408c`](https://github.com/nlz25/PFD_Agent/commit/01d408cb948ab5b0d91aad71b34a38d5e65ff0a5))

- Restore the loop design
  ([`de4202f`](https://github.com/nlz25/PFD_Agent/commit/de4202fa7fe7ab5b1a8e180e58f8d7463641a2f2))

- Restructure agent team and skills
  ([`b731197`](https://github.com/nlz25/PFD_Agent/commit/b73119738956c52396fa86ae34d344321acbe358))

- Update skill refreshing mechanism
  ([`de46787`](https://github.com/nlz25/PFD_Agent/commit/de46787ecda86a8492b3d9480daf73216c9af472))

- Update web interface
  ([`492db55`](https://github.com/nlz25/PFD_Agent/commit/492db5576c9af5cab7f5ce300d7b89657d3bc372))

### Features

- Add workspace control
  ([`b9ff30f`](https://github.com/nlz25/PFD_Agent/commit/b9ff30f8ec469a5da11a1c69e7173a0e19a2c661))


## v1.0.21 (2026-04-13)

### Bug Fixes

- Skills update
  ([`5316dbf`](https://github.com/nlz25/PFD_Agent/commit/5316dbf6aa493a825111caece6fdf087728ea83f))


## v1.0.20 (2026-04-09)

### Bug Fixes

- Update the web interface
  ([`e6978a2`](https://github.com/nlz25/PFD_Agent/commit/e6978a24168617629bd72ede2a6cf075feb72f62))


## v1.0.19 (2026-04-03)

### Bug Fixes

- Display function call in UI
  ([`eb6f47f`](https://github.com/nlz25/PFD_Agent/commit/eb6f47fe515dcea8eaeaff14f68b4f2032caf2e1))


## v1.0.18 (2026-04-02)

### Bug Fixes

- Add trajectory.py
  ([`c20c055`](https://github.com/nlz25/PFD_Agent/commit/c20c0557686a98452c115f4f723d6cf4bc829e89))


## v1.0.17 (2026-03-31)

### Bug Fixes

- Add clear_current_skill function
  ([`836c32c`](https://github.com/nlz25/PFD_Agent/commit/836c32ce7d7b9058749d073465d1b0f75c316ef3))

- Add trajectory
  ([`f4e41c4`](https://github.com/nlz25/PFD_Agent/commit/f4e41c43bcee60965f9e98f7e0e6d87e38940344))

- Rescope crystal structure skill to quests
  ([`f876ec1`](https://github.com/nlz25/PFD_Agent/commit/f876ec1065abbe8ec64b1042bd834491ef829c1a))


## v1.0.16 (2026-03-31)

### Bug Fixes

- Add atomic_structure skill
  ([`ac855d0`](https://github.com/nlz25/PFD_Agent/commit/ac855d0468537fb82f4ec6a0c46f7e2454ecddfc))


## v1.0.15 (2026-03-30)

### Bug Fixes

- Update basic tools
  ([`06d1653`](https://github.com/nlz25/PFD_Agent/commit/06d1653e684efb0faaf3416f39aca22414edbe0d))

- Update UI
  ([`b4647b6`](https://github.com/nlz25/PFD_Agent/commit/b4647b6f7365dacefd7dd49935c4248860fb2628))


## v1.0.14 (2026-03-29)

### Bug Fixes

- Add util_tools
  ([`5811cb6`](https://github.com/nlz25/PFD_Agent/commit/5811cb68e56b15c6bf0fd37df82cada804ac1ceb))


## v1.0.13 (2026-03-29)

### Bug Fixes

- Remove subagents
  ([`c976184`](https://github.com/nlz25/PFD_Agent/commit/c97618430917251bb5df3c1eff9974ffc50147f5))


## v1.0.12 (2026-03-27)

### Bug Fixes

- Demolish execution agent
  ([`1b0afe4`](https://github.com/nlz25/PFD_Agent/commit/1b0afe447e92b79a1e5f6be56faa02ed42a0d447))

- Update central agent
  ([`e4f640e`](https://github.com/nlz25/PFD_Agent/commit/e4f640ee996c933b48ffa7f7faa2d2ebb7427589))

- Update memory
  ([`9d4adbb`](https://github.com/nlz25/PFD_Agent/commit/9d4adbb7e148e03c3374b5a8f048f68b534da7fe))


## v1.0.11 (2026-03-25)

### Bug Fixes

- Add deempd and dpdispatcher skill
  ([`4acee84`](https://github.com/nlz25/PFD_Agent/commit/4acee84daa48ff458ae6175cf0044f648b860876))

- Error handling in execution agent
  ([`419a8c3`](https://github.com/nlz25/PFD_Agent/commit/419a8c37de7f4bc40d628a57599ba4a879b5f07a))

- Update the memory setting
  ([`a962070`](https://github.com/nlz25/PFD_Agent/commit/a9620705ed47c7c53dcef7818bda12c2cd80558d))


## v1.0.10 (2026-03-24)

### Bug Fixes

- Skillization of database toolset
  ([`f533b61`](https://github.com/nlz25/PFD_Agent/commit/f533b61777eb0dcdf19f5b50d6bcd3eaf014ab68))

- Skillize crystal structure tools
  ([`bfdf048`](https://github.com/nlz25/PFD_Agent/commit/bfdf048b3eed6aa84b1645d2d73809147d22e9ef))

- Skillize VASP
  ([`be28867`](https://github.com/nlz25/PFD_Agent/commit/be2886759df6023d2ef409362588b5886f489eea))

- Update README
  ([`8d4c3d7`](https://github.com/nlz25/PFD_Agent/commit/8d4c3d722798a0d2f965a39561723bb2384e9311))


## v1.0.9 (2026-03-23)

### Bug Fixes

- Add a new release branch for packaging
  ([`22d3e4f`](https://github.com/nlz25/PFD_Agent/commit/22d3e4f9848c6b3838bb215690ed2d2e0daad38f))


## v1.0.8 (2026-03-20)

### Bug Fixes

- Update workspace path
  ([`11530ef`](https://github.com/nlz25/PFD_Agent/commit/11530efb88d52ffa6fe241c2eded2a2561a25f45))


## v1.0.7 (2026-03-19)

### Bug Fixes

- Issues with mattergen tool
  ([`05cd41e`](https://github.com/nlz25/PFD_Agent/commit/05cd41ebaa0afca5337e3848912a0c2f06f523a9))

- Simplify summarization and planning agent
  ([`4a4fb38`](https://github.com/nlz25/PFD_Agent/commit/4a4fb3810b820061946548216089b7c4bda2cc8c))


## v1.0.6 (2026-03-19)

### Bug Fixes

- Issues with dynamic tool loading
  ([`d7ab70f`](https://github.com/nlz25/PFD_Agent/commit/d7ab70f7f42f7eed2ed5d6006a1c7523d5cea05e))

- Issues with mcp tools
  ([`75b803c`](https://github.com/nlz25/PFD_Agent/commit/75b803c52614132b58d6d9b16719f5cb175e3ac4))

- Restructure the planning agent
  ([`db6df8c`](https://github.com/nlz25/PFD_Agent/commit/db6df8c514bfd19bb5f4c8ff7d2a2651f7e24fe4))


## v1.0.5 (2026-03-18)

### Bug Fixes

- Issues with endless execution loop
  ([`fa79a02`](https://github.com/nlz25/PFD_Agent/commit/fa79a02a13ce7cf409ef39aa5b3fb0f07db305d9))


## v1.0.4 (2026-03-16)

### Bug Fixes

- Issues with skill load
  ([`7201ce6`](https://github.com/nlz25/PFD_Agent/commit/7201ce67765ad314495541fa55b9b96bcdbc6d44))

### Chores

- Update README
  ([`8a08dc9`](https://github.com/nlz25/PFD_Agent/commit/8a08dc98034296912d02b5ea47b53a4a35b3861e))


## v1.0.3 (2026-03-16)

### Bug Fixes

- Move workspace to home directory
  ([`8f318b5`](https://github.com/nlz25/PFD_Agent/commit/8f318b550fe2a65724be50d3f0bf9f2f92200584))

- Replacing all sub-agents with skills
  ([`88e19ee`](https://github.com/nlz25/PFD_Agent/commit/88e19ee7cf3305b22b843d54c765217cdeeb0a15))

- Restructure guides and skills
  ([`f5282d9`](https://github.com/nlz25/PFD_Agent/commit/f5282d968421834a7f9ce75d9d28f521ca744fa2))

### Chores

- Remove depreacted tools
  ([`710ca1f`](https://github.com/nlz25/PFD_Agent/commit/710ca1fe3bdcb52b612775d495d526a0f22cacb5))

- Remove redundant files
  ([`d5d203e`](https://github.com/nlz25/PFD_Agent/commit/d5d203ee7589a0dfd0c9e5bfd87f131f8e939075))


## v1.0.2 (2026-03-11)

### Bug Fixes

- Add logic for force break
  ([`140a368`](https://github.com/nlz25/PFD_Agent/commit/140a368232a8c2f0df7c0e8cee9f69e4416315bf))

- Dynamic compactionn within invocatiion
  ([`74bb0ea`](https://github.com/nlz25/PFD_Agent/commit/74bb0eabc53b1e4a00bea4c1d476625474a2aa28))

- Improved workflow
  ([`83c53f2`](https://github.com/nlz25/PFD_Agent/commit/83c53f28d5dd5baaf1b915c83c02b1618a31ba8f))

- New dflow decorators for dpa tools
  ([`d5c0999`](https://github.com/nlz25/PFD_Agent/commit/d5c0999801d6cdffa0d86f4159dd9d7c99eff670))

- Update README for dpa tools
  ([`eb25840`](https://github.com/nlz25/PFD_Agent/commit/eb25840be3029bf8175bf6bd70d4dc5c54f451d9))


## v1.0.1 (2026-03-08)

### Bug Fixes

- Add README for database tool
  ([`0c67099`](https://github.com/nlz25/PFD_Agent/commit/0c670999c1c8378d6b321c104e425203e8d1488e))

- Add self-check for planning agent
  ([`98c5db1`](https://github.com/nlz25/PFD_Agent/commit/98c5db1b62ef510b7505a5388782ec5ce74c6a5e))

- Issues with DPA finetuning setting
  ([`6e96414`](https://github.com/nlz25/PFD_Agent/commit/6e96414ac16db3bcefd76fbd2ea48b59f029b5a1))


## v1.0.0 (2026-03-06)

- Initial Release
