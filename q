TERMINATOR(1)                                                           TERMINATOR(1)

NNAAMMEE
       Terminator - Multiple GNOME terminals in one window

SSYYNNOOPPSSIISS
       tteerrmmiinnaattoorr [_o_p_t_i_o_n_s]

DDEESSCCRRIIPPTTIIOONN
       This manual page documents TTeerrmmiinnaattoorr, a terminal emulator application.

       TTeerrmmiinnaattoorr  is  a program that allows users to set up flexible arrangements of
       GNOME terminals. It is aimed at those who normally arrange lots  of  terminals
       near each other, but don't want to use a frame based window manager.

OOPPTTIIOONNSS
       This  program  follow  the  usual  GNU  command line syntax, with long options
       starting with two dashes (`-').  A summary of options is included below.

       --hh,, ----hheellpp
              Show summary of options

       --vv,, ----vveerrssiioonn
              Show the version of the Terminator installation

       --mm,, ----mmaaxxiimmiissee
              Start with a maximised window

       --ff,, ----ffuullllssccrreeeenn
              Start with a fullscreen window

       --bb,, ----bboorrddeerrlleessss
              Instruct the window manager not to  render  borders/decoration  on  the
              Terminator window (this works well with -m)

       --HH,, ----hhiiddddeenn
              Hide  the  Terminator  window by default. Its visibility can be toggled
              with the hhiiddee__wwiinnddooww keyboard shortcut (Ctrl-Shift-Alt-a by default)

       --TT,, ----ttiittllee
              Force the Terminator window to use a specific name rather than updating
              it dynamically based on the wishes of the child shell.

       ----ggeeoommeettrryy==GGEEOOMMEETTRRYY
              Specifies  the  preferred size and position of Terminator's window; see
              X(7).

       --ee,, ----ccoommmmaanndd==CCOOMMMMAANNDD
              Runs the specified command instead of your  default  shell  or  profile
              specified command. Note: if Terminator is launched as x-terminal-emula‐
              tor -e behaves like -x, and the longform becomes --execute2=COMMAND

       --xx,, ----eexxeeccuuttee CCOOMMMMAANNDD [[AARRGGSS]]
              Runs tthhee rreesstt ooff tthhee ccoommmmaanndd lliinnee instead of your default shell or pro‐
              file specified command.

       ----wwoorrkkiinngg--ddiirreeccttoorryy==DDIIRR
              Set the terminal's working directory

       --gg,, ----ccoonnffiigg FFIILLEE
              Use the specified FILE for configuration

       --rr,, ----rroollee==RROOLLEE
              Set a custom WM_WINDOW_ROLE property on the window

       --cc,, ----ccllaassssnnaammee==CCLLAASSSSNNAAMMEE
              Set a custom name (WM_CLASS) property on the window

       --ll,, ----llaayyoouutt==LLAAYYOOUUTT
              Start  Terminator with a specific layout. The argument here is the name
              of a saved layout.

       --ss,, ----sseelleecctt--llaayyoouutt==LLAAYYOOUUTT
              Open the layout launcher window instead of the normal terminal.

       --pp,, ----pprrooffiillee==PPRROOFFIILLEE
              Use a different profile as the default

       --ii,, ----iiccoonn==FFOORRCCEEDDIICCOONN
              Set a custom icon for the window (by file or name)

       --uu,, ----nnoo--ddbbuuss
              Disable DBus

       --dd,, ----ddeebbuugg
              Enable debugging output (please use this when reporting bugs). This can
              be specified twice to enable a built-in python debugging server.

       ----ddeebbuugg--ccllaasssseess==DDEEBBUUGG__CCLLAASSSSEESS
              If  this  is specified as a comma separated list, debugging output will
              only be printed from the specified classes.

       ----ddeebbuugg--mmeetthhooddss==DDEEBBUUGG__MMEETTHHOODDSS
              If this is specified as a comma separated list, debugging  output  will
              only  be  printed from the specified functions. If this is specified in
              addition to --debug-classes, only the intersection  of  the  two  lists
              will be displayed

       ----nneeww--ttaabb
              If  this  is  specified and Terminator is already running, DBus will be
              used to spawn a new tab in the first Terminator window.

KKEEYYBBIINNDDIINNGGSS
       The following default keybindings can be used to control Terminator:

       FF11     Launches the full HTML manual.

   CCrreeaattiioonn && DDeessttrruuccttiioonn
       The following items relate to creating and destroying terminals.

       CCttrrll++SShhiifftt++OO
              Split terminals Hoorizontally.

       CCttrrll++SShhiifftt++EE
              Split terminals Veertically.

       CCttrrll++SShhiifftt++TT
              Open new ttab.

       CCttrrll++SShhiifftt++II
              Open a new window. (Note: Unlike in previous releases, this  window  is
              part of the same Terminator process.)

       SSuuppeerr++II
              Spawn a new Terminator process.

       AAlltt++LL  Open llayout launcher.

       CCttrrll++SShhiifftt++WW
              Close the current terminal.

       CCttrrll++SShhiifftt++QQ
              Close the current window.

   NNaavviiggaattiioonn
       The following items relate to moving between and around terminals.

       AAlltt++UUpp Move to the terminal aabboovvee the current one.

       AAlltt++DDoowwnn
              Move to the terminal bbeellooww the current one.

       AAlltt++LLeefftt
              Move to the terminal lleefftt ooff the current one.

       AAlltt++RRiigghhtt
              Move to the terminal rriigghhtt ooff the current one.

       CCttrrll++PPaaggeeDDoowwnn
              Move to next Tab.

       CCttrrll++PPaaggeeUUpp
              Move to previous Tab.

       CCttrrll++SShhiifftt++NN oorr CCttrrll++TTaabb
              Move to nnext terminal within the same tab, use Ctrl+PageDown to move to
              the next tab.  If ccyyccllee__tteerrmm__ttaabb is FFaallssee, cycle within  the  same  tab
              will be disabled.

       CCttrrll++SShhiifftt++PP oorr CCttrrll++SShhiifftt++TTaabb
              Move  to pprevious terminal within the same tab, use Ctrl+PageUp to move
              to the previous tab.  If ccyyccllee__tteerrmm__ttaabb is FFaallssee, cycle within the same
              tab will be disabled.

   OOrrggaanniissaattiioonn
       The following items relate to arranging and resizing terminals.

       CCttrrll++SShhiifftt++RRiigghhtt
              Move parent dragbar RRiigghhtt.

       CCttrrll++SShhiifftt++LLeefftt
              Move parent dragbar LLeefftt.

       CCttrrll++SShhiifftt++UUpp
              Move parent dragbar UUpp.

       CCttrrll++SShhiifftt++DDoowwnn
              Move parent dragbar DDoowwnn.

       SSuuppeerr++RR
              RRotate terminals clockwise.

       SSuuppeerr++SShhiifftt++RR
              RRotate terminals counter-clockwise.

       DDrraagg aanndd DDrroopp
              The  layout can be modified by moving terminals with Drag and Drop.  To
              start dragging a terminal, click and hold on  its  titlebar.   Alterna‐
              tively,  hold  down CCttrrll, click and hold the rriigghhtt mouse button.  Then,
              ****RReelleeaassee CCttrrll****. You can now drag the terminal to  the  point  in  the
              layout  you  would like it to be.  The zone where the terminal would be
              inserted will be highlighted.

       CCttrrll++SShhiifftt++PPaaggeeDDoowwnn
              Swap tab position with next Tab.

       CCttrrll++SShhiifftt++PPaaggeeUUpp
              Swap tab position with previous Tab.

   MMiisscceellllaanneeoouuss
       The following items relate to miscellaneous terminal related functions.

       CCttrrll++SShhiifftt++CC
              Copy selected text to clipboard.

       CCttrrll++SShhiifftt++VV
              Paste clipboard text.

       CCttrrll++SShhiifftt++SS
              Hide/Show SScrollbar.

       CCttrrll++SShhiifftt++FF
              Search within terminal scrollback.

       CCttrrll++SShhiifftt++RR
              Reset terminal state.

       CCttrrll++SShhiifftt++GG
              Reset terminal state and clear window.

       CCttrrll++PPlluuss ((++))
              Increase font size. NNoottee:: This may require you to press shift,  depend‐
              ing on your keyboard.

       CCttrrll++MMiinnuuss ((--))
              Decrease  font size. NNoottee:: This may require you to press shift, depend‐
              ing on your keyboard.

       CCttrrll++ZZeerroo ((00))
              Restore font size to original setting.

       CCttrrll++AAlltt++WW
              Rename window title.

       CCttrrll++AAlltt++AA
              Rename tab title.

       CCttrrll++AAlltt++XX
              Rename terminal title.

       SSuuppeerr++11
              Insert terminal number, i.e. 1 to 12.

       SSuuppeerr++00
              Insert padded terminal number, i.e. 01 to 12.

   GGrroouuppiinngg && BBrrooaaddccaassttiinngg
       The following items relate to helping to focus on a specific terminal.

       FF1111    Toggle window to fullscreen.

       CCttrrll++SShhiifftt++XX
              Toggle between showing all terminals and only showing the  current  one
              (maximise).

       CCttrrll++SShhiifftt++ZZ
              Toggle  between showing all terminals and only showing a scaled version
              of the current one (zoom).

       CCttrrll++SShhiifftt++AAlltt++AA
              Hide the initial window. Note that this is a global  binding,  and  can
              only be bound once.

       The following items relate to grouping and broadcasting.

       SSuuppeerr++TT
              Group  all  terminals  in the current tab so input sent to one of them,
              goes to all terminals in the current tab.

       SSuuppeerr++SShhiifftt++TT
              Remove grouping from all terminals in the current tab.

       SSuuppeerr++GG
              Group all terminals so that any input sent to one of them, goes to  all
              of them.

       SSuuppeerr++SShhiifftt++GG
              Remove grouping from all terminals.

       AAlltt++AA  Broadcast to All terminals.

       AAlltt++GG  Broadcast to Grouped terminals.

       AAlltt++OO  Broadcast Off.

       Most of these keybindings are changeable in the Preferences.

SSEEEE AALLSSOO
       tteerrmmiinnaattoorr__ccoonnffiigg((55))

AAUUTTHHOORR
       Terminator was written by Chris Jones <cmsj@tenshu.net> and others.

       This manual page was written by Chris Jones <cmsj@tenshu.net> and others.

                                     Jan 5, 2008                        TERMINATOR(1)
